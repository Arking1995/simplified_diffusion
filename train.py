import time
from parser import BaseOptions
from transformers import set_seed, AutoTokenizer
from functools import partial
import json, torch, os
import numpy as np
from utils import *

from text_datasets import load_data_text
from all_utils.resample import create_named_schedule_sampler

from all_utils.test_util import get_weights, compute_logp
from all_utils.rounding import load_models, load_tokenizer
from functools import partial
from train_util import TrainLoop
import wandb
import all_utils.logger as logger

def get_mapping_func(args, diffusion, data):
    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    args.checkpoint_path, extra_args=args)
    model3 = get_weights(model2, args)
    print(model3, model3.weight.requires_grad)
    mapping_func = partial(compute_logp, args, model3.cuda())
    diffusion.mapping_func = mapping_func
    return mapping_func


if __name__ == '__main__':

    args_updated = BaseOptions().parse()   # get training options
    args = create_argparser(vars(args_updated)).parse_args()

    # print(args)
    # if 'num_channels' in vars(args):
    #     print(vars(args)['num_channels'])
    # print(model_and_diffusion_defaults().keys())

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to('cuda:0')
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'the parameter count is {pytorch_total_params}')

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    wandb.login(key='51c4a548a8a34d48ce63eeb0159960f30c973d74')
    wandb.init(
        project=os.getenv("LayoutDiffusion", "diffusion_lm"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)


    print('load data', '*'*50)
    
    if args.use_bert_tokenizer == 'yes':
        rev_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        rev_tokenizer = None

    model22 = None

    data = load_data_text(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        data_args = args,
        task_mode=args.modality,
        padding_mode=args.padding_mode, #block, pad
        load_vocab=rev_tokenizer,
        model=model22,
    )
    next(data)
    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    args.checkpoint_path, extra_args=args)

    if args.modality == 'book' or args.use_bert_tokenizer == 'yes':
        rev_tokenizer = tokenizer # BERT tokenizer BPE.
    else:
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

    data_valid = load_data_text(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        data_args=args,
        task_mode=args.modality,
        padding_mode=args.padding_mode,  # block, pad
        split='valid',
        load_vocab=rev_tokenizer,
        model=model2,
    )


    get_mapping_func(args, diffusion, data)

    # import pdb
    # pdb.set_trace()

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()