import time
from options.train_options import TrainOptions
from data import CustomDataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import os
from util import util
import numpy as np
import torch
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from skimage import data, io

def load_pretrained_model(model, opt):
    model.load_networks(epoch='latest')  # Use 'latest' suffix to load latest model

def save_metrics_to_file(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(f"Epoch: {metrics['epoch']}, MSE: {metrics['MSE']}, PSNR: {metrics['PSNR']}, SSIM: {metrics['SSIM']}\n")

def calculateMean(vars):
    return sum(vars) / len(vars) if vars else 0

def save_img(path, img):
    fold, name = os.path.split(path)
    os.makedirs(fold, exist_ok=True)
    io.imsave(path, img)
    

def evaluateModel(model, opt, test_dataset):
    model.netG.eval()
    eval_path = os.path.join(opt.checkpoints_dir, 'evaluate', 'Eval.csv')
    eval_results_fstr = open(eval_path, 'w')
    eval_results_fstr.write("Image Path, Mask Mean, MSE, PSNR, SSIM\n")  # 添加SSIM列标题
    eval_results = {
        'mask': [], 'mse': [], 'psnr': [], 'fmse': [], 'ssim': [],
        'low': {'mse': [], 'psnr': [], 'fmse': [], 'ssim': []},
        'mid': {'mse': [], 'psnr': [], 'fmse': [], 'ssim': []},
        'high': {'mse': [], 'psnr': [], 'fmse': [], 'ssim': []}
    }

    for i, data in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        model.set_input(data)  
        #flops, params = profile(model, inputs=data)
        #print(f"FLOPs: {flops}, Params: {params}")
        model.test()  
        visuals = model.get_current_visuals()  
        output = visuals['attentioned']
        real = visuals['real']

        for i_img in range(real.size(0)):
            gt, pred = real[i_img:i_img+1], output[i_img:i_img+1]
            fore_ratio = data['mask'][i_img].mean().item()
            fore_nums = data['mask'][i_img].sum().item()

            pred_np = util.tensor2im(pred)
            gt_np = util.tensor2im(gt)

            mse_score_op = mean_squared_error(pred_np, gt_np)
            psnr_score_op = peak_signal_noise_ratio(gt_np, pred_np, data_range=255)
            ssim_score_op = ssim(gt_np, pred_np, data_range=255, channel_axis=-1)
            fmse_score_op = mse_score_op * 256 * 256 / fore_nums if fore_nums > 0 else 0

            img_path = data['img_path'][i_img]
            basename, imagename = os.path.split(img_path)
            basename = basename.split('/')[-2]
            save_dir = os.path.join(opt.checkpoints_dir, 'evaluate', 'results', basename)
            os.makedirs(save_dir, exist_ok=True)
            save_img(os.path.join(save_dir, imagename.split('.')[0] + '.png'), pred_np)

            # 添加指标
            eval_results['mse'].append(mse_score_op)
            eval_results['psnr'].append(psnr_score_op)
            eval_results['ssim'].append(ssim_score_op)
            eval_results['fmse'].append(fmse_score_op)         
            eval_results['mask'].append(fore_ratio)
            
            # 根据前景比例分类
            if fore_ratio <= 0.05:
                eval_results['low']['mse'].append(mse_score_op)
                eval_results['low']['psnr'].append(psnr_score_op)
                eval_results['low']['ssim'].append(ssim_score_op)
                eval_results['low']['fmse'].append(fmse_score_op)
            elif fore_ratio <= 0.15:
                eval_results['mid']['mse'].append(mse_score_op)
                eval_results['mid']['psnr'].append(psnr_score_op)
                eval_results['mid']['ssim'].append(ssim_score_op)
                eval_results['mid']['fmse'].append(fmse_score_op)
            else:
                eval_results['high']['mse'].append(mse_score_op)
                eval_results['high']['psnr'].append(psnr_score_op)
                eval_results['high']['ssim'].append(ssim_score_op)
                eval_results['high']['fmse'].append(fmse_score_op)

            eval_results_fstr.writelines('%s,%.3f,%.3f,%.3f,%.4f\n' % (
                data['img_path'][i_img], fore_ratio, mse_score_op, psnr_score_op, ssim_score_op))

    eval_results_fstr.flush()
    eval_results_fstr.close()

    # 计算平均值
    all_mse, all_psnr, all_fmse, all_ssim = (
        calculateMean(eval_results['mse']),
        calculateMean(eval_results['psnr']),
        calculateMean(eval_results['fmse']),
        calculateMean(eval_results['ssim'])
    )
    low_mse, low_psnr, low_fmse, low_ssim = (
        calculateMean(eval_results['low']['mse']),
        calculateMean(eval_results['low']['psnr']),
        calculateMean(eval_results['low']['fmse']),
        calculateMean(eval_results['low']['ssim'])
    )
    mid_mse, mid_psnr, mid_fmse, mid_ssim = (
        calculateMean(eval_results['mid']['mse']),
        calculateMean(eval_results['mid']['psnr']),
        calculateMean(eval_results['mid']['fmse']),
        calculateMean(eval_results['mid']['ssim'])
    )
    high_mse, high_psnr, high_fmse, high_ssim = (
        calculateMean(eval_results['high']['mse']),
        calculateMean(eval_results['high']['psnr']),
        calculateMean(eval_results['high']['fmse']),
        calculateMean(eval_results['high']['ssim'])
    )

    print(f'MSE: {all_mse:.3f}, PSNR: {all_psnr:.3f}, fMSE: {all_fmse:.3f}, SSIM: {all_ssim:.4f}')
    print(f'Foreground (0-0.05) - MSE: {low_mse:.3f}, PSNR: {low_psnr:.3f}, fMSE: {low_fmse:.3f}, SSIM: {low_ssim:.4f}')
    print(f'Foreground (0.05-0.15) - MSE: {mid_mse:.3f}, PSNR: {mid_psnr:.3f}, fMSE: {mid_fmse:.3f}, SSIM: {mid_ssim:.4f}')
    print(f'Foreground (0.15-1.0) - MSE: {high_mse:.3f}, PSNR: {high_psnr:.3f}, fMSE: {high_fmse:.3f}, SSIM: {high_ssim:.4f}')

    model.netG.train()
    return all_mse, all_psnr, all_ssim

if __name__ == '__main__':
    opt = TrainOptions().parse()  
    test_dataset = CustomDataset(opt, is_for_train=False)
    test_dataloader = test_dataset.load_data()
    opt.isTrain = False
    model = create_model(opt)  
    model.setup(opt)  
    load_pretrained_model(model, opt)

    evaluateModel(model, opt, test_dataloader)
    print('Evaluation completed.')
