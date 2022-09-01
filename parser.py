import sys
import os
import argparse
import subprocess

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser(description='training args.')
        parser.add_argument('--experiment', type=str, default='random', help='no-rep=gpt2gen, no-zipfs, has-rep=regular, rm-window-rep')

        parser.add_argument('--model_arch', type=str, default='transformer', help='')
        parser.add_argument('--modality', type=str, default='layout', help='')
        parser.add_argument('--noise_schedule', type=str, default='sqrt', help='')
        parser.add_argument('--loss_type', type=str, default='Lsimple', help='')
        parser.add_argument('--dropout', type=str, default='0.1', help='')
        parser.add_argument('--weight_decay', type=str, default=0.0, help='')

        parser.add_argument('--image_size', type=int, default=8, help='')
        parser.add_argument('--hidden_size', type=int, default=128, help='')
        parser.add_argument('--in_channel', type=int, default=128, help='')
        parser.add_argument('--out_channel', type=int, default=128, help='')
        parser.add_argument('--m', type=int, default=3, help='')
        parser.add_argument('--k', type=int, default=32, help='')
        parser.add_argument('--lr_anneal_steps', type=int, default=200000, help='')
        parser.add_argument('--num_res_blocks', type=int, default=2, help='')

        parser.add_argument('--lr', type=float, default=1e-04, help='')
        parser.add_argument('--bsz', type=int, default=64, help='')
        parser.add_argument('--diff_steps', type=int, default=2000, help='')
        parser.add_argument('--padding_mode', type=str, default='pad', help='')
        parser.add_argument('--seed', type=int, default=102, help='') # old is 42

        parser.add_argument('--notes', type=str, default=None, help='')
        parser.add_argument('--submit', type=str, default='no', help='')
        parser.add_argument('--use_big', type=str, default='no', help='')
        parser.add_argument('--app', type=str, default='', help='')
        parser.add_argument('--folder_name', type=str)


        parser.add_argument('--predict_xstart', type=bool, default = True, help=" ")
        parser.add_argument('--training_mode', type=str, default='e2e')
        parser.add_argument('--vocab_size', type=int, default=1857)
        parser.add_argument('--e2e_train', type=str, default='../datasets/e2e_data')      
        parser.add_argument('--use_kl', type=bool, default = False)
        parser.add_argument('--learn_sigma', type=bool, default=False)    

        parser.add_argument('--batch_size', type=int, default=64)    
        parser.add_argument('--checkpoint_path', type=str, default="diff_test")

        args = parser.parse_args()
        folder_name = 'diffusion_models/'
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        dir_path = os.path.join(folder_name, args.checkpoint_path)

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)   

        args.checkpoint_path = dir_path
        return args


    def parse(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            args = self.initialize(parser)

        return args
