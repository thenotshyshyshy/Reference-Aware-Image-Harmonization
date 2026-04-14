import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from .fMSE import MaskWeightedMSE
from .diffusion import diffusion
from .encoder import Encoder_lr, Encoder_gt, denoise
#from .contra import contraLoss
#from thop import profile
import psutil
import os
class HDNetModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['comp', 'real', 'output', 'mask', 'real_f', 'fake_f', 'bg', 'attentioned']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.relu = nn.ReLU()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = MaskWeightedMSE(100)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.g_lr_ratio, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        self.encoder = Encoder_gt(feats=64, scale=4).cuda()
        self.condition = Encoder_lr(feats=64, scale=4).cuda()
        self.denoise = denoise(feats=64, timesteps=4).cuda()
        self.netGD = diffusion.DDPM(denoise=self.denoise, condition=self.condition ,feats=64, timesteps = 4).cuda()
        #self.contrastive_loss_fn = contraLoss(self.encoder, self.condition)
    
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = self.comp
        if self.opt.input_nc == 4:
            self.inputs = torch.cat([self.inputs, self.mask], 1)  # channel-wise concatenation
        self.real_f = self.real * self.mask
        self.bg = self.real * (1 - self.mask)
    
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def contrastive_loss(anchor, positive, negatives, margin=0.2):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dists = [F.pairwise_distance(anchor, neg) for neg in negatives]
        loss = F.relu(pos_dist - torch.stack(neg_dists).min() + margin)
        return loss.mean()

    def forward(self):
        #self.output = self.netG(self.inputs, self.mask)
        #self.fake_f = self.output * self.mask
        #self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
        #self.harmonized = self.attentioned 
        if self.isTrain:
            self.freeze_module(self.encoder) 
            self.cdp = self.encoder(self.comp,self.real)
            self.cdp_diff = self.netGD(self.comp,self.cdp)
            #self.cdp_diff = self.netGD(self.comp)
            self.output = self.netG(self.inputs,self.mask,self.cdp_diff)
            #device = torch.device('cuda:1')
            #self.netG = self.netG.to(device)
            #flops, params = profile(self.netG, inputs=(dummy_inputs, dummy_mask, dummy_cdp_diff))
            #print(f"FLOPs: {flops}, Params: {params}")

            self.fake_f = self.output * self.mask
            self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
            self.harmonized = self.attentioned 
            
            # 打印统计信息
            #self.print_memory_usage()
            
            return self.harmonized
        else:
        
            self.cdp_diff = self.netGD(self.comp)
            self.output = self.netG(self.inputs, self.mask, self.cdp_diff)
            self.fake_f = self.output * self.mask
            self.attentioned = self.output * self.mask + self.inputs[:,:3,:,:] * (1 - self.mask)
            self.harmonized = self.attentioned 
            #self.print_memory_usage()
            
            return self.harmonized
    
    def print_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 ** 2  # 单位：MB
        print(f"当前进程使用的内存: {mem:.2f} MB")

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.attentioned, self.real, self.mask) * self.opt.lambda_L1
        #self.contrastive_loss_value = self.contrastive_loss_fn(self.comp, self.real, self.mask)
        #self.loss_G = self.loss_G_L1+0.001*self.contrastive_loss_value
        self.loss_diff = F.l1_loss(self.cdp, self.cdp_diff)
        self.loss_G = self.loss_G_L1+0.01*self.loss_diff    
        #self.loss_G = self.loss_G_L1+0.01*self.loss_diff+0.1*self.noise_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
         # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        torch.cuda.empty_cache()