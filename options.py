import argparse
import os
from util import util
import torch

class TestOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--result_dir', required=True, help='path to save images')
        parser.add_argument('--batchSize',type=int,default=1,help='input batch size')
        parser.add_argument('--loadSizeX',type=int,default=256,help='scale images to this size')
        parser.add_argument('--loadSizeY',type=int,default=256,help='scale images to this size')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--image_feature', type=int, default=512, help='the max channels for image features')
        parser.add_argument('--fineSize', type=int, default=256, help='crop the image to')
        parser.add_argument('--norm', type=str, default='batch', help='batch normalization or instance normalization')
        parser.add_argument('--activation', type=str, default='PReLU', help='ReLu, LeakyReLU, PReLU, or SELU')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='model initialization [normal|xavier|kaiming]')
        parser.add_argument('--name',type=str,default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--nThreads',default=1,type=int,help='# threads for loading data')
        parser.add_argument('--checkpoints_dir',type=str,default='D:/myda/checkpoints/',help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--isTrain',type=bool, default=False,help='no dropout for the generator')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        parser.add_argument('--display_freq', type=int, default=100,
                            help='frequency of showing training results on screen')
        parser.add_argument('--show_freq', type=int, default=100, help='frequency of showing training results on plot')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000000000000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--learn_residual', type=bool, default=False,
                            help='if specified, model would learn only the residual to the input')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()

        # opt.expr_name = opt.src_dataset + '2' + opt.tgt_dataset + '_' + opt.model
        # # process opt.suffix
        # if opt.suffix:
        # 	suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        # 	opt.expr_name = opt.expr_name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to images')
        parser.add_argument('--batchSize',type=int,default=1,help='input batch size')
        parser.add_argument('--loadSizeX',type=int,default=512,help='scale images to this size')
        parser.add_argument('--loadSizeY',type=int,default=512,help='scale images to this size')
        parser.add_argument('--output_nc',type=int,default=3,help='# of output image channels')
        parser.add_argument('--ngf',type=int,default=64,help='# of gen filters in first conv layer')
        parser.add_argument('--ndf',type=int,default=64,help='# of discrim filters in first conv layer')
        parser.add_argument('--image_feature',type=int,default=512,help='the max channels for image features')
        parser.add_argument('--fineSize',type=int,default=256,help='crop the image to')
        parser.add_argument('--norm',type=str,default='batch',help='batch normalization or instance normalization')
        parser.add_argument('--activation',type=str,default='PReLU',help='ReLu, LeakyReLU, PReLU, or SELU')
        parser.add_argument('--init_type',type=str,default='kaiming',help='model initialization [normal|xavier|kaiming]')
        parser.add_argument('--drop_rate',type=float,default=0,help='# of drop rate')
        parser.add_argument('--gan_type',type=str,default='wgan-gp',
            help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
        parser.add_argument('--name',type=str,default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--learn_residual', type=bool, default=False,
                            help='if specified, model would learn only the residual to the input')
        parser.add_argument('--nThreads',default=1,type=int,help='# threads for loading data')
        parser.add_argument('--checkpoints_dir',type=str,default='D:/myda/checkpoints/',help='models are saved here')
        parser.add_argument('--serial_batches',action='store_true',
            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize',type=int,default=256,help='display window size')
        parser.add_argument('--display_id',type=int, default=1,help='window id of the web display')
        parser.add_argument('--display_port',type=int,default=8097,help='visdom port of the web display')
        parser.add_argument('--no_dropout',action='store_true',help='no dropout for the generator')
        parser.add_argument('--isTrain',type=bool, default=False,help='no dropout for the generator')
        parser.add_argument('--phase',default='train',help='phase')
        parser.add_argument('--resize_or_crop',type=str,default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip',action='store_true',help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--pool_size',type=int,default=50,help='the size of image buffer that stores previously generated images')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--lr_depth', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--wd', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--momentum', type=float, default=0.9, help='initial learning rate for adam')
        parser.add_argument('--display_freq', type=int, default=100,
                            help='frequency of showing training results on screen')
        parser.add_argument('--show_freq', type=int, default=100, help='frequency of showing training results on plot')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000000000000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true',
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--no_lsgan', type=bool, default=False,
                            help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--patience', type=int, default=5, help='tbd')
        parser.add_argument('--lambda_D', type=float, default=1.0, help='tbd')
        parser.add_argument('--ihz_w', type=float, default=0.1, help='tbd')
        parser.add_argument('--lambda_a', type=float, default=0.003, help='tbd')
        parser.add_argument('--lambda_consistency',type=float, default=0.003, help='tbd')
        parser.add_argument('--lambda_coherency',type=float, default=0.003, help='tbd')
        parser.add_argument('--lambda_beta', type=float, default=0.003, help='tbd')
        parser.add_argument('--lambda_depth', type=float, default=0.001, help='tbd')
        parser.add_argument('--lambda_mse_hazy', type=float, default=3, help='tbd')
        parser.add_argument('--lambda_mse_hazy2', type=float, default=0.1, help='tbd')
        parser.add_argument('--lambda_l1_depth', type=float, default=1, help='tbd')
        parser.add_argument('--lambda_l1_depth2', type=float, default=0.1, help='tbd')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()

        # opt.expr_name = opt.src_dataset + '2' + opt.tgt_dataset + '_' + opt.model
        # # process opt.suffix
        # if opt.suffix:
        # 	suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        # 	opt.expr_name = opt.expr_name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
