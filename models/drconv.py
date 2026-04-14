import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable, Function

import matplotlib.pyplot as plt

def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk,  **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk,  **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)



def create_multicolor_guide_mask(guide_mask_tensor, colors=None):
    """创建多颜色的guide_mask可视化，类似图中的效果"""
    if colors is None:
        # 默认颜色：红、绿、蓝对应mask, inv_mask, learned_attention
        colors = [(1.0, 0.2, 0.2),    # 红色 - mask (W0)
                 (0.2, 1.0, 0.2),    # 绿色 - inv_mask (W1) 
                 (0.2, 0.2, 1.0)]    # 蓝色 - learned_attention (W2)
    
    print(f"Input guide_mask_tensor shape: {guide_mask_tensor.shape}")
    
    # 处理输入tensor的维度 - 更详细的维度处理
    original_shape = guide_mask_tensor.shape
    
    # 处理各种可能的输入形状
    if len(original_shape) == 5:  # B x 3 x 1 x H x W
        guide_mask = guide_mask_tensor[0]  # 取第一个batch: 3 x 1 x H x W
        guide_mask = guide_mask.squeeze(1)  # 去掉中间维度: 3 x H x W
    elif len(original_shape) == 4:  
        if original_shape[1] == 1:  # 3 x 1 x H x W
            guide_mask = guide_mask_tensor.squeeze(1)  # 3 x H x W
        elif original_shape[0] == 1:  # 1 x 3 x H x W
            guide_mask = guide_mask_tensor[0]  # 3 x H x W
        else:  # 3 x H x W (已经是正确形状)
            guide_mask = guide_mask_tensor
    elif len(original_shape) == 3:  # 3 x H x W
        guide_mask = guide_mask_tensor
    else:
        raise ValueError(f"Unexpected guide_mask shape: {original_shape}")
    
    print(f"Processed guide_mask shape: {guide_mask.shape}")
    
    # 确保guide_mask是3 x H x W的形状
    if len(guide_mask.shape) != 3 or guide_mask.shape[0] != 3:
        raise ValueError(f"After processing, guide_mask should be (3, H, W), but got {guide_mask.shape}")
    
    H, W = guide_mask.shape[-2:]
    print(f"Height: {H}, Width: {W}")
    
    # 转换为numpy并归一化
    masks = []
    for i in range(3):
        mask = guide_mask[i].detach().cpu().numpy()
        # 归一化到[0,1]
        if mask.max() > mask.min():
            mask = (mask - mask.min()) / (mask.max() - mask.min())
        else:
            mask = np.zeros_like(mask)
        masks.append(mask)
        print(f"Mask {i} shape: {mask.shape}, min: {mask.min():.3f}, max: {mask.max():.3f}")
    
    # 创建RGB图像 - 加权叠加版本
    rgb_image = np.zeros((H, W, 3))
    for i, (mask, color) in enumerate(zip(masks, colors)):
        for c in range(3):  # RGB三个通道
            rgb_image[:, :, c] += mask * color[c]
    
    # 归一化到[0,1]范围
    rgb_image = np.clip(rgb_image, 0, 1)
    print(f"RGB image shape: {rgb_image.shape}")
    
    # 创建离散版本 - argmax版本（更接近原图效果）
    weight_argmax = torch.argmax(guide_mask, dim=0)  # H x W
    weight_argmax_np = weight_argmax.detach().cpu().numpy()
    print(f"Weight argmax shape: {weight_argmax_np.shape}")
    
    # 创建离散颜色图 - 确保输出形状正确
    discrete_colors = np.array(colors)  # 3 x 3 (3个颜色，每个3个通道)
    print(f"Discrete colors shape: {discrete_colors.shape}")
    
    # 正确的索引方式：对每个像素位置，根据argmax结果选择对应颜色
    colored_argmax = np.zeros((H, W, 3))
    for h in range(H):
        for w in range(W):
            region_idx = weight_argmax_np[h, w]
            colored_argmax[h, w] = discrete_colors[region_idx]
    
    print(f"Colored argmax final shape: {colored_argmax.shape}")
    
    return rgb_image, colored_argmax, masks

def create_guide_mask_legend(colors, labels=None):
    """创建guide_mask的图例"""
    if labels is None:
        labels = ['Mask (W₀)', 'Inv-Mask (W₁)', 'Learned (W₂)']
    
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # 创建颜色块
    for i, (color, label) in enumerate(zip(colors, labels)):
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(i+0.5, 0.5, label, ha='center', va='center', 
               fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Guide Mask Components', fontsize=14, fontweight='bold', pad=20)
    
    return fig

def visualize_multicolor_guide_mask(guide_mask_tensor, save_path=None):
    """完整的多颜色guide_mask可视化"""
    import os
    
    # 如果没有提供路径，使用当前工作目录的绝对路径
    if save_path is None:
        save_path = os.path.abspath("./guide_mask_visualization")
    else:
        save_path = os.path.abspath(save_path)
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    colors = [(1.0, 0.2, 0.2),    # 红色
             (0.2, 1.0, 0.2),    # 绿色
             (0.2, 0.2, 1.0)]    # 蓝色
    
    try:
        # 生成多颜色guide_mask
        rgb_guide_mask, argmax_guide_mask, individual_masks = create_multicolor_guide_mask(
            guide_mask_tensor, colors
        )
        
        # 验证输出形状
        print(f"RGB guide mask shape: {rgb_guide_mask.shape}")
        print(f"Argmax guide mask shape: {argmax_guide_mask.shape}")
        
        # 创建图例
        legend_fig = create_guide_mask_legend(colors)
        
        # 主要的多颜色可视化
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：单独的注意力组件（灰度）
        titles = ['Mask (W₀)', 'Inv-Mask (W₁)', 'Learned (W₂)']
        for i, (mask, title) in enumerate(zip(individual_masks, titles)):
            if i < len(axes[0]):  # 防止索引越界
                axes[0, i].imshow(mask, cmap='gray')
                axes[0, i].set_title(f'{title} - Grayscale', fontsize=12)
                axes[0, i].axis('off')
        
        # 第二行：彩色表示
        for i, (mask, color, title) in enumerate(zip(individual_masks, colors, titles)):
            if i < len(axes[1]):  # 防止索引越界
                colored_mask = np.zeros((*mask.shape, 3))
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c]
                axes[1, i].imshow(colored_mask)
                axes[1, i].set_title(f'{title} - Colored', fontsize=12)
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # 组合的多颜色guide_mask
        fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 组合的RGB图像（渐变叠加）
        axes[0].imshow(rgb_guide_mask)
        axes[0].set_title('Combined Guide Mask\n(Weighted RGB Overlay)', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 离散颜色版本（argmax，类似原图）- 确保形状正确
        if len(argmax_guide_mask.shape) == 3:  # H x W x 3
            axes[1].imshow(argmax_guide_mask)
        else:
            print(f"Warning: argmax_guide_mask has unexpected shape: {argmax_guide_mask.shape}")
            # 如果形状不对，显示一个占位符
            placeholder = np.zeros((128, 128, 3))
            axes[1].imshow(placeholder)
            
        axes[1].set_title('Dominant Region per Pixel\n(Discrete Colors - Like Original Figure)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # 保存图片 - 使用绝对路径
        legend_path = os.path.join(save_path, "guide_mask_legend.png")
        components_path = os.path.join(save_path, "guide_mask_components.png") 
        combined_path = os.path.join(save_path, "guide_mask_combined.png")
        
        legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
        fig1.savefig(components_path, dpi=300, bbox_inches='tight')
        fig2.savefig(combined_path, dpi=300, bbox_inches='tight')
        
        print(f"图片已保存到:")
        print(f"图例: {legend_path}")
        print(f"组件: {components_path}")
        print(f"组合: {combined_path}")
        
        # 关闭图形以释放内存
        plt.close('all')
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print(f"Guide mask tensor shape: {guide_mask_tensor.shape}")
        import traceback
        traceback.print_exc()
    
    return rgb_guide_mask, argmax_guide_mask


class DRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, region_num=3, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = 3

        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, **kwargs)
        
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        kernel = self.conv_kernel(input)
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H

        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        mask = mask.unsqueeze(1) 
        guide_feature = self.conv_guide(input)
        #guide_feature = torch.relu(guide_feature)
        #guide_feature=torch.sigmoid(guide_feature)
        guide_feature=guide_feature.unsqueeze(1) 
        #guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
        inv_msak = 1 - mask
        guide_mask = torch.cat((mask,inv_msak,guide_feature), 1)
        #visualize_multicolor_guide_mask(guide_mask.squeeze(0) )

        output = torch.sum(output * guide_mask, dim=1)
        return output


