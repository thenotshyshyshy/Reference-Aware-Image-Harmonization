import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage import data, io

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def calculateMean(vars):
    return sum(vars) / len(vars)

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)


def evaluateModel(model, opt, test_dataset):
    model.netG.eval()
    
    eval_results = {'mse': [], 'psnr': [], 'fmse':[], 'ssim':[]}

    for i, data in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        model.set_input(data)  # unpack data from data loader
        model.test()  # inference
        visuals = model.get_current_visuals()  # get image results
        output = visuals['attentioned']
        real = visuals['real']

        for i_img in range(real.size(0)):
            gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
            fore_nums = data['mask'][i_img].sum().item()
            mse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt))
            psnr_score_op = peak_signal_noise_ratio(util.tensor2im(gt), util.tensor2im(pred), data_range=255)
            fmse_score_op = mean_squared_error(util.tensor2im(pred), util.tensor2im(gt)) * 256 * 256 / fore_nums
            ssim_score = ssim(util.tensor2im(pred), util.tensor2im(gt), data_range=255, channel_axis=-1)
            
            pred_rgb = util.tensor2im(pred)
            img_path = data['img_path'][i_img]
            basename, imagename = os.path.split(img_path)
            basename = basename.split('/')[-2]
            #save_img(os.path.join('evaluate', str(epoch_number), 'results',basename, imagename.split('.')[0] + '.png'), pred_rgb)
            save_dir = os.path.join(weights_path,'results', basename)
            os.makedirs(save_dir, exist_ok=True)
            save_img(os.path.join(save_dir, imagename.split('.')[0] + '.png'), pred_rgb)
            
            # update calculator
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['fmse'].append(fmse_score_op)         
            eval_results['ssim'].append(ssim_score) 
            #eval_results['mask'].append(data['mask'][i_img].mean().item())
            #eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f\n' % (data['img_path'][i_img], eval_results['mask'][-1],mse_score_op, psnr_score_op))
        if i + 1 % 100 == 0:
            # print('%d images have been processed' % (i + 1))
            eval_results_fstr.flush()
    eval_results_fstr.flush()
    eval_results_fstr.close()
    
    all_mse, all_psnr, all_fmse, all_ssim = calculateMean(eval_results['mse']), calculateMean(eval_results['psnr']),  calculateMean(eval_results['fmse']),  calculateMean(eval_results['ssim'])
    
    print('MSE:%.3f, PSNR:%.3f, fMSE:%.3f, SSIM:%.3f' % (all_mse, all_psnr, all_fmse, all_ssim))
    model.netG.train()
    return all_mse, all_psnr

if __name__ == '__main__':
    # setup_seed(6)
    opt = TrainOptions().parse()   # get training options
    
    # check if multiple GPUs are detected
    if len(opt.gpu_ids) > 1:
        print('WARNING: Multiple GPUs detected, this could lead to inefficiency')
    
    test_dataset = CustomDataset(opt, is_for_train=False)
    test_dataset_size = len(test_dataset)
    print('The number of testing images = %d' % test_dataset_size)
    
    test_dataloader = test_dataset.load_data()

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # Load pretrained weights from a specific path
    weights_path = "/opt/data/private/HDNet/checkpoints2/experiment_train/latest"
    print('Loading model weights from %s' % weights_path)
    model.load_networks(weights_path)

    epoch_mse, epoch_psnr = evaluateModel(model, opt, test_dataloader)
    print("Evaluation Results - MSE: %.3f, PSNR: %.3f" % (epoch_mse, epoch_psnr))

