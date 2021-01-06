from model.base_model import BaseModel
import torch
import itertools
from model import losses
from util.image_pool import ImagePool
from model import networks
from model.L1_TVLoss import L1_TVLoss_Charbonnier

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
                'D_A',
                'D_b',
                'D_d',
                'G_A',
                'G_b',
                'G_d',
                'b',
                'A',
                'd',
                'mse_depth',
                'mse_hzy']
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
            self.A_discriminator = networks.define_D(
                3,
                opt.ndf,
                'n_layers',
                opt.ndf,
                norm=opt.norm,
                use_sigmoid=True,
                gpu_ids=opt.gpu_ids,
                use_parallel=False)
            self.beta_discriminator = networks.define_D(
                3,
                opt.ndf,
                'n_layers',
                opt.ndf,
                norm=opt.norm,
                use_sigmoid=True,
                gpu_ids=opt.gpu_ids,
                use_parallel=False)
            self.d_discriminator = networks.define_D(
                1,
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
                params=itertools.chain(
                    self.A_discriminator.parameters(),
                    self.beta_discriminator.parameters(),
                    self.d_discriminator.parameters()),
                lr=opt.lr_d,
                betas=(
                    0.9,
                    0.99))
            self.ganloss = losses.GANLoss(use_ls=not opt.no_lsgan).to(self.device)
            self.mseloss = torch.nn.MSELoss()
            self.TVloss = L1_TVLoss_Charbonnier()
            self.nonlinearity = torch.nn.ReLU()
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_b_pool = ImagePool(opt.pool_size)
            self.fake_d_pool = ImagePool(opt.pool_size)

    def set_input(self, input):

        if self.isTrain:
            input_A = input['A']
            input_B = input['B']
            input_C = input['C']
            input_E = input['E']
            self.nyu_gt = input_A.to(self.device)
            self.ihz_gt = input_C.to(self.device)
            self.nyu_depth = input_B.to(self.device)
            self.ihz_hzy = input_E.to(self.device)
            self.image_paths = input['A_paths']
        else:
            self.img = input['A'].to(self.device)

    def forward(self):
        if self.isTrain:
            pass

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, l):
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.ganloss(pred_real, True)
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
        l_d = self.opt.lambda_depth
        l_dA = self.opt.lambda_da
        l_db = self.opt.lambda_dbeta
        l_dd = self.opt.lambda_ddepth
        l_mse_hzy = self.opt.lambda_mse_hazy
        l_mse_depth = self.opt.lambda_mse_depth
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
        self.loss_A = (self.TVloss(self.nyu_Amap) + self.TVloss(self.ihz_Amap)) / 2.0
        self.loss_b = (self.TVloss(self.nyu_bmap) + self.TVloss(self.ihz_bmap)) / 2.0
        self.loss_d = (self.TVloss(self.nyu_dmap) + self.TVloss(self.ihz_dmap)) / 2.0
        self.loss_G_A = self.ganloss(self.A_discriminator(self.nyu_Amap), True)
        self.loss_G_b = self.ganloss(self.beta_discriminator(self.nyu_bmap), True)
        self.loss_G_d = self.ganloss(self.d_discriminator(self.nyu_dmap), True)
        self.loss_mse_depth = self.mseloss(self.nyu_depth, self.nyu_dmap)
        self.loss_mse_hzy = self.mseloss(self.ihz_gen, self.ihz_hzy)
        self.loss_g = self.loss_A*l_A + self.loss_b*l_b + self.loss_d*l_d + \
            self.loss_mse_depth*l_mse_depth + self.loss_mse_hzy*l_mse_hzy + \
            self.loss_G_A * l_dA + self.loss_G_b* l_db + self.loss_G_d * l_dd
        self.loss_g.backward()

    def backward_D_A(self):
        l_da = self.opt.lambda_da
        nyu_A = self.fake_A_pool.query(self.nyu_Amap)
        self.loss_D_A = self.backward_D_basic(
            self.A_discriminator, nyu_A, self.ihz_Amap, l_da)

    def backward_D_beta(self):
        l_db = self.opt.lambda_dbeta
        nyu_b = self.fake_b_pool.query(self.nyu_bmap)
        self.loss_D_b = self.backward_D_basic(
            self.beta_discriminator, nyu_b, self.ihz_bmap,l_db)

    def backward_D_depth(self):
        l_dd = self.opt.lambda_ddepth
        nyu_d = self.fake_d_pool.query(self.nyu_dmap)
        self.loss_D_d = self.backward_D_basic(
            self.d_discriminator, nyu_d, self.ihz_dmap,l_dd)

    def optimize_parameters(self):
        self.forward()
        self.backward_G()
        self.optimizer_G.step()
        self.backward_D_A()
        self.backward_D_beta()
        self.backward_D_depth()
        self.optimizer_D.step()
