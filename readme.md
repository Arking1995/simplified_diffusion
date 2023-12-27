# Diffusion-Based Document Layout Generation
Official Pytorch Implementation of "Diffusion-Based Document Layout Generation"

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-41676-7_21) | [Project Page](https://arking1995.github.io/DIFF_DOC/)

This repo contains codes for single GPU training for the diffusion backbone in the paper. The original codes are referred to [Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM).

To run the training code, simply "python train.py", default arguments will make it run on Publaynet dataset in the repo.

## Environment
```
Pytorch==1.8.0
cudatoolkit==11.1
transformers==4.20.1
huggingface-hub==0.8.1
```

## How to train your model
Sample command:
```
python train.py {arguments below}
```

Argument list locates in "parser.py". Basic modifications across different datasets are:
```
    --batch_size  "defaultly 64, it only uses about 8GB memory on my local GPU"
    --lr           "learning rate"
    --lr_anneal_steps   "how many steps you wanna train"
    --diff_steps  "how many diffusion steps are used in this training"
    --in_channel      "The length of each token embedding vector, also the input and output embedding vector length of the transformer used in diffusion process"   in/output is shape [batch_size *  max_sequence_length * in/out_channel]
    --out_channel     "The same size as in_channel, diffussion is an end-to-end process"
    --vocab_size      "Typically, it should be 256 + category_number + 4 (start, end, pad, unknown token). But since dataset has some noise input value, you can setup a safe larger number to make it run"
    --checkpoint_path   "The checkpoints will be output to './diffusion_models/{checkpoint_path}', "./diffusion_models" will be generated automatically.
    --pln_train       "The path to training dataset folder, 'train.txt' and 'val.txt' should be in that folder"
    --padding_mode    "default='pad', padding the sequence, or "block" "
    --noise_schedule   "scheduled noise added to diffusion process, it can be 'linear, cosine, sqrt, trunc_cos, trunc_lin, pw_lin'. default=sqrt "
    --num_samples      'The number of intermediate output samples.'
    --save_interval       'Save interval by steps'
```
For better debugging, model initilization in "utils.py", trainer in "train_utils.py", dataset loading in "text_dataset.py", models are in "all_utils/gaussian_diffusion.py", 

All files under ./all_utlis can be directly used without change for now.

Current codes will automatically upload training curve to my wandb account, if you don't like it, just change the crediential of wandb.init() in "train.py"

## How to inference
```
python text_sample.py --model_path {model path ".pt"} --batch_size 1 --num_samples 1000 --top_p -1.0 --out_dir {output directory}
```





