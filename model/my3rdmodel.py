from model.base_model import BaseModel
import torch
import itertools
from collections import OrderedDict
from model import losses
from util.image_pool import ImagePool
from util import util
from model import networks
from model.L1_TVLoss import *
from torch.autograd import Variable
import os
from model.fcrn import ResNet
import numpy as np

def create_model(opt):
    if(opt.isTrain==True):
        instance = Mymodel()
    else:
        instance = MyTestmodel()
    instance.initialize(opt=opt)
    print("model [%s] was created" % (instance.name()))
    return instance


class MyTestmodel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.visual_names = ['nyu_gt', 'nyu_depth', 'nyu_Amap', 'nyu_bmap', 'nyu_dmap', 'nyu_gen']
        self.input_A = self.Tensor(opt.batchSize, 3, opt.loadSizeX, opt.loadSizeY)

        self.A_generator = networks.define_G(
            3,
            3,
            opt.ngf,
            'resnet_9blocks',
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            use_parallel=False,
            learn_residual=opt.learn_residual)
        self.beta_generator = networks.define_G(
            3,
            3,
            opt.ngf,
            'resnet_9blocks',
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            use_parallel=False,
            learn_residual=opt.learn_residual)
        self.d_generator = networks.define_Gen_d(output_size=(self.opt.fineSize, self.opt.fineSize), layer=50,
                                                 init_type=self.opt.init_type, gpu_ids=self.gpu_ids)
        self.load_network(self.A_generator, 'latest_2_A_generator.pth')
        self.load_network(self.beta_generator, 'latest_2_beta_generator.pth')
        self.load_network(self.d_generator, 'latest_2_d_generator.pth')
        self.synthesizer = networks.Synthesizer()


        print('---------- Networks initialized -------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.A_name = input['A_name']
        self.A_size = input['A_size']
        temp = self.input_A.clone()
        temp.resize_(input_A.size()).copy_(input_A)
        self.input_A = temp
        self.image_paths = input['A_path']

    def test(self):
        self.nyu_gt = Variable(self.input_A, volatile = True)
        self.nyu_Amap = self.A_generator.forward(self.nyu_gt)
        self.nyu_bmap = self.beta_generator.forward(self.nyu_gt)
        self.nyu_dmap = self.d_generator.forward(self.nyu_gt)
        self.nyu_gen = self.synthesizer(
            self.nyu_gt,
            self.nyu_Amap,
            self.nyu_bmap,
            self.nyu_dmap)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def save_result(self, save_path):
        name = self.A_name
        size = self.A_size
        nyu_Amap = util.tensor2im(self.nyu_Amap.data)
        nyu_bmap = util.tensor2im(self.nyu_bmap.data)
        nyu_dmap = util.tensor2im(self.nyu_dmap.data)
        nyu_gen = util.tensor2im(self.nyu_gen.data)
        util.save_sized_image(nyu_Amap, save_path+'A/' + name[0], size)
        util.save_sized_image(nyu_bmap, save_path + 'beta/' + name[0], size)
        util.save_sized_image(nyu_dmap, save_path + 'depth/' + name[0], size)
        util.save_sized_image(nyu_gen, save_path + 'result/' + name[0], size)


class Mymodel(BaseModel):
    def name(self):
        return 'Mymodel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.stage=0
        self.loss_names = [
            'D',
            'G',
            'b',
            'A',
            'd',
            'l1_depth',
            'mse_hzy',
            'consistency',
            'coherency']
        self.visual_names = [
            'nyu_Amap',
            'nyu_bmap',
            'nyu_dmap',
            'ihz_Amap',
            'ihz_bmap',
            'ihz_dmap',
            'ihz_gen',
            'nyu_gen']
        self.visual_names += ['nyu_gt', 'nyu_depth','ihz_gt', 'ihz_hzy']

        self.model_names+=['A_generator', 'beta_generator', 'd_generator', 'discriminator']

        self.A_generator = networks.define_G(
            3,
            3,
            opt.ngf,
            'resnet_9blocks',
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            use_parallel=False,
            learn_residual=opt.learn_residual)
        self.beta_generator = networks.define_G(
            3,
            3,
            opt.ngf,
            'resnet_9blocks',
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            use_parallel=False,
            learn_residual=opt.learn_residual)
        self.d_generator = networks.define_Gen_d(output_size=(self.opt.fineSize,self.opt.fineSize),layer=50,
                                                 init_type= self.opt.init_type,gpu_ids=self.gpu_ids)
        self.maskedl1loss = losses.MaskedL1Loss()
        self.Huberloss = losses.HuberLoss()
        self.synthesizer = networks.Synthesizer()
        if self.isTrain:
            self.discriminator = networks.define_D(
                3,
                opt.ndf,
                'n_layers',
                opt.ndf,
                norm=opt.norm,
                use_sigmoid=True,
                gpu_ids=opt.gpu_ids,
                use_parallel=False)
            self.optimizer_D = torch.optim.Adam(
                params=self.discriminator.parameters(),
                lr=opt.lr_d,
                betas=(
                    0.9,
                    0.99))
            self.ganloss = losses.GANLoss(use_ls=not opt.no_lsgan).to(self.device)
            self.mseloss = torch.nn.MSELoss()
            self.TVloss = L1_TVLoss_Charbonnier()
            self.consistancyloss = Inter_Channel_Consistancy()
            self.coherencyloss= Inter_Channel_Coherence()
            self.nonlinearity = torch.nn.ReLU()
            self.fake_nyu_pool = ImagePool(opt.pool_size)

    def set_stage(self, i):
        self.stage = i
        if(i==0):
            self.l_A = 0
            self.l_b = 0
            self.l_depth = 0
            self.l_D = 0
            self.l_mse_hzy = 0
            self.l_l1_depth = self.opt.lambda_l1_depth
            self.l_consistency = 0
            self.l_coherency = 0
            train_params = [{'params': self.d_generator.get_1x_lr_params(), 'lr': self.opt.lr_depth},
                            {'params': self.d_generator.get_10x_lr_params(), 'lr': self.opt.lr_depth * 10}]
            self.optimizer_G = torch.optim.Adam(
                params=train_params,
                lr=self.opt.lr_depth,
                betas=(0.9,0.99))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, 'min', patience=self.opt.patience)
        elif(i==1):
            self.l_A = self.opt.lambda_a
            self.l_b = self.opt.lambda_beta
            self.l_depth = 0
            self.l_D = self.opt.lambda_D
            self.l_mse_hzy = self.opt.lambda_mse_hazy
            self.l_l1_depth = 0
            self.l_consistency = self.opt.lambda_consistency
            self.l_coherency = self.opt.lambda_coherency
            self.optimizer_G = torch.optim.Adam(
                params=itertools.chain(
                    self.A_generator.parameters(),
                    self.beta_generator.parameters()),
                lr=self.opt.lr_g,
                betas=(
                    0.9,
                    0.99))
        elif(i==2):
            self.l_A = self.opt.lambda_a
            self.l_b = self.opt.lambda_beta
            self.l_depth = self.opt.lambda_depth
            self.l_D = self.opt.lambda_D
            self.l_mse_hzy = self.opt.lambda_mse_hazy2
            self.l_l1_depth = self.opt.lambda_l1_depth2
            self.l_consistency = self.opt.lambda_consistency
            self.l_coherency = self.opt.lambda_coherency
            self.optimizer_G = torch.optim.Adam(
                params=itertools.chain(
                    self.A_generator.parameters(),
                    self.beta_generator.parameters(),
                    self.d_generator.parameters()),
                lr=self.opt.lr_g,
                betas=(
                    0.9,
                    0.99))
        else:
            raise RuntimeError("STAGE MUST BE 0,1 OR 2")

    def set_pretrain(self):
        if(self.opt.pretrained_depth):
            path='D:/myda/pretrained/model_best.pth'
            self.set_stage(1)


    def set_input(self, input):

        if self.isTrain:
            input_A = input['A']
            input_B = input['B']
            input_C = input['C']
            input_D = input['D']
            input_E = input['E']
            self.nyu_gt = input_A.to(self.device)
            self.ihz_gt = input_C.to(self.device)
            self.nyu_depth = input_B.to(self.device)
            self.real_hzy =input_D.to(self.device)
            self.ihz_hzy = input_E.to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.img = input['A'].to(self.device)

    def forward(self):
        if self.isTrain:
            pass

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real1, real2 , fake, l, w):
        # Real
        pred_real1 = netD(real1.detach())
        loss_D_real1 = self.ganloss(pred_real1, True)
        pred_real2 = netD(real2.detach())
        loss_D_real2 = self.ganloss(pred_real2, True)
        loss_D_real = loss_D_real1 * w + loss_D_real2 *(1-w)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.ganloss(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * l
        # backward
        loss_D.backward()
        return loss_D

    def backward_G(self):
        self.nyu_A = self.A_generator(self.nyu_gt)
        self.nyu_Amap = torch.ones_like(self.nyu_A)*torch.mean(self.nyu_A)
        self.nyu_bmap = self.beta_generator(self.nyu_gt)
        self.nyu_dmap = self.d_generator(self.nyu_gt)
        self.ihz_A = self.A_generator(self.ihz_gt)
        self.ihz_Amap = torch.ones_like(self.ihz_A)*torch.mean(self.ihz_A)
        self.ihz_bmap = self.beta_generator(self.ihz_gt)
        self.ihz_dmap = self.d_generator(self.ihz_gt)
        self.nyu_gen = self.synthesizer(
            self.nyu_gt,
            self.nyu_Amap,
            self.nyu_bmap,
            self.nyu_dmap)
        self.ihz_gen = self.synthesizer(
            self.ihz_gt,
            self.ihz_Amap,
            self.ihz_bmap,
            self.ihz_dmap)
        self.loss_A = self.l_A*(self.TVloss(self.nyu_Amap) + self.TVloss(self.ihz_Amap)) / 2.0
        self.loss_b = self.l_b*(self.TVloss(self.nyu_bmap) + self.TVloss(self.ihz_bmap)) / 2.0
        self.loss_d = self.l_depth*(self.TVloss(self.nyu_dmap) + self.TVloss(self.ihz_dmap)) / 2.0
        self.loss_G = (self.ganloss(self.discriminator(self.ihz_gen),True)
                        + self.ganloss(self.discriminator(self.nyu_gen),True))/2
        self.loss_mse_hzy = self.mseloss(self.ihz_gen, self.ihz_hzy)*self.l_mse_hzy
        self.loss_consistency = self.l_consistency*(self.consistancyloss(self.nyu_Amap)+self.consistancyloss(self.ihz_Amap))/2.0
        self.loss_coherency = self.l_coherency*(self.coherencyloss(self.nyu_bmap) + self.coherencyloss(self.ihz_bmap))/2.0
        self.loss_l1_depth = self.maskedl1loss(self.nyu_dmap, self.nyu_depth) * self.l_l1_depth
        self.loss_g = self.loss_A + self.loss_b + self.loss_d + self.loss_mse_hzy + \
            self.loss_G + self.loss_consistency + self.loss_coherency + self.loss_l1_depth
        self.loss_g.backward()

    def backward_D(self):
        l_D = self.opt.lambda_D
        ihz_gen = self.fake_nyu_pool.query(self.ihz_gen)
        nyu_gen = self.fake_nyu_pool.query(self.nyu_gen)
        self.loss_D = self.backward_D_basic(self.discriminator, self.ihz_hzy, self.real_hzy, self.nyu_gen, l_D, self.opt.ihz_w)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.discriminator], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.discriminator], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


