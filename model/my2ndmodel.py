from model.base_model import BaseModel
import torch
import itertools
from model import losses
from util.image_pool import ImagePool
from model import networks
from model.L1_TVLoss import *

def create_model(opt):
    instance = Mymodel()
    instance.initialize(opt=opt)
    print("model [%s] was created" % (instance.name()))
    return instance

class Mymodel(BaseModel):
    def name(self):
        return 'Mymodel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if self.isTrain:
            self.loss_names = [
                'D',
                'G',
                'b',
                'A',
                'd',
                'mse_depth',
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
        else:
            self.visual_names = ['gt', 'amap', 'bmap', 'dmap', 'syn']

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
        self.d_generator = networks.define_G(
            3,
            1,
            opt.ngf,
            'resnet_9blocks',
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            gpu_ids=self.gpu_ids,
            use_parallel=False,
            learn_residual=opt.learn_residual)
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
            self.optimizer_G = torch.optim.Adam(
                params=itertools.chain(
                    self.A_generator.parameters(),
                    self.beta_generator.parameters(),
                    self.d_generator.parameters()),
                lr=opt.lr_g,
                betas=(
                    0.9,
                    0.99))
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
        l_A = self.opt.lambda_a
        l_b = self.opt.lambda_beta
        l_depth = self.opt.lambda_depth
        l_D = self.opt.lambda_D
        l_mse_hzy = self.opt.lambda_mse_hazy
        l_mse_depth = self.opt.lambda_mse_depth
        l_consistency = self.opt.lambda_consistency
        l_coherency = self.opt.lambda_coherency
        self.nyu_Amap = self.A_generator(self.nyu_gt)
        self.nyu_bmap = self.beta_generator(self.nyu_gt)
        self.nyu_dmap = self.d_generator(self.nyu_gt)
        self.ihz_Amap = self.A_generator(self.ihz_gt)
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
        self.loss_A = l_A*(self.TVloss(self.nyu_Amap) + self.TVloss(self.ihz_Amap)) / 2.0
        self.loss_b = l_b*(self.TVloss(self.nyu_bmap) + self.TVloss(self.ihz_bmap)) / 2.0
        self.loss_d = l_depth*(self.TVloss(self.nyu_dmap) + self.TVloss(self.ihz_dmap)) / 2.0
        self.loss_G = self.ganloss(self.discriminator(self.ihz_gen),True)\
                        + self.ganloss(self.discriminator(self.nyu_gen),True)
        self.loss_mse_depth =self.mseloss(self.nyu_depth, self.nyu_dmap)*l_mse_depth
        self.loss_mse_hzy = self.mseloss(self.ihz_gen, self.ihz_hzy)*l_mse_hzy
        self.loss_consistency = l_consistency*(self.consistancyloss(self.nyu_Amap)+self.consistancyloss(self.ihz_Amap))/2.0
        self.loss_coherency = l_coherency*(self.coherencyloss(self.nyu_bmap) + self.coherencyloss(self.ihz_bmap))/2.0
        self.loss_g = self.loss_A + self.loss_b + self.loss_d + \
            self.loss_mse_depth + self.loss_mse_hzy + \
            self.loss_G + self.loss_consistency + self.loss_coherency
        self.loss_g.backward()

    def backward_D(self):
        l_D = self.opt.lambda_D
        ihz_gen = self.fake_nyu_pool.query(self.ihz_gen)
        nyu_gen = self.fake_nyu_pool.query(self.nyu_gen)
        self.loss_D = self.backward_D_basic(self.discriminator, nyu_gen, self.ihz_hzy, self.real_hzy, l_D, self.opt.ihz_w)


    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
