"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from all_utils.rounding import rounding_func, load_models, load_tokenizer

from all_utils.test_util import get_weights, denoised_fn_round

import all_utils.logger as logger
from functools import partial
from utils import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    set_seed(101)
    args = create_argparser().parse_args()

    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    args.diffusion_steps = 100 #500  # DEBUG
    print("exp: ", args.experiment)
    print("exp_mode: ", args.experiment_mode)

    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print('model path: ', args.model_path)
    model = th.nn.DataParallel(model, list(range(1))).to('cuda:0')
    model.load_state_dict(th.load(args.model_path, map_location='cuda:0'))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    print('model arch: ', args.model_arch)
    # model.to('cuda')
    model.eval() # DEBUG


    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model2.weight = th.nn.Parameter(model.module.word_embedding.weight.clone().cpu())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    print(args.num_samples, args.batch_size)
    model3 = get_weights(model2, args)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # print('sample_fn: ', sample_fn)

        if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
            sample_shape = (args.batch_size * args.mbr_sample, args.image_size ** 2, args.in_channel)
        else:
            sample_shape = (args.batch_size, args.image_size ** 2, args.in_channel)

        print("sample_shape: ", sample_shape)
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,
            denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
            model_kwargs=model_kwargs,
            top_p =args.top_p,
        )
        print('in sample: ', sample)

        print("sample_shape.shape: ", sample.shape)


        # gathered_samples = [th.zeros_like(sample)]
        all_images.extend([sample.cpu().numpy()])

        logger.log(f"created {len(all_images) * args.batch_size} samples")

    print('all_image: ', all_images)
    arr = np.concatenate(all_images, axis=0)
    print(arr.shape, 'full shape')
    arr = arr[: args.num_samples * args.mbr_sample]
    print('arr: ', arr)


    if diffusion.training_mode.startswith('e2e'):
        word_lst_e2e = []
        print('decoding for e2e', )
        print(arr.shape)
        x_t = th.tensor(arr).cuda()
        if args.model_arch == 'conv-unet':
            reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
        else:
            reshaped_x_t = x_t

        print('reshaped_x_t: ',reshaped_x_t)
        logits = model.module.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        print('logits: ', logits)
        cands = th.topk(logits, k=1, dim=-1)
        sample = cands.indices
        tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
        for seq in cands.indices:
            if isinstance(tokenizer, dict):
                tokens = " ".join([tokenizer[x[0].item()] for x in seq])
            else:
                tokens = tokenizer.decode(seq.squeeze(-1))
            word_lst_e2e.append(tokens)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    
    
    shape_str = "x".join([str(x) for x in arr.shape])
    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npz")
    # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")

    if args.verbose == 'yes':
        logger.log('decode by rounding. ')
        print('load_models')
        if diffusion.training_mode.startswith('e2e'):
            word_lst = word_lst_e2e
            print('e2e: ', word_lst)
        else:
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                           os.path.split(args.model_path)[0])
            print('rounding')
            word_lst = rounding_func(args.experiment, arr, model, tokenizer,
                                     emb_scale_factor=args.emb_scale_factor)

        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.txt")
        fout = open(out_path2, 'w')

        for (xx) in zip( word_lst):
            # print('---' * 30)
            # print(tokenizer.decode(gg.tolist()))
            # print('xx' * 30)
            print(xx[0], file=fout)
            # print('---' * 30)
        fout.close()
        print(f'written the decoded output to {out_path2}')

        ##############
        out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.json")
        fout = open(out_path2, 'w')
        for (xx) in zip(word_lst):
            print(json.dumps(xx), file=fout)
        fout.close()
        print(f'written the decoded output to {out_path2}')




def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,#10000,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen"
    )
    text_defaults = dict(modality='layout',
                         dataset_name='PublayNet',
                         dataset_config_name='layout-PublayNet',
                         model_name_or_path='/opt/data/liuhe95/simplified_diffusion/pretrained_model',
                         experiment='e2e-tgt', model_arch='transformer',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
