import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import log
from ref_func import generate_ide_fn
from nerf_helpers import linear_to_srgb
from gpu_mem_track import MemTracker


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            rgb = F.sigmoid(rgb)
            outputs = torch.cat([rgb, alpha], -1)

        return outputs  

class RefNeRF(nn.Module):
    def __init__(self, sh_max_level = 5, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], bottle_neck_dim=128):
        """
        D: 深度，多少层网络
        W: 网络内的宽度
        input_ch: xyz的宽度
        input_ch_views: direction的宽度
        output_ch: 这个参数仅在 use_viewdirs=False的时候会被使用
        skips: 类似resnet的残差连接，表明在第几层进行连接
        """
        super(RefNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.bottle_neck_dim = bottle_neck_dim
        self.ide = generate_ide_fn(sh_max_level)
        self.ide_dim = ((1 << sh_max_level) - 1 + sh_max_level) << 1

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.view_dim = 1 + self.bottle_neck_dim + self.ide_dim
        self.views_linears = nn.ModuleList(
            [nn.Linear(self.view_dim, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.view_dim, W) for i in
                                             range(D - 1)])
        
        self.rough_linear = nn.Linear(W, 1)

        self.bottle_neck_linear = nn.Linear(W, bottle_neck_dim)

        self.norm_color_tint_linear = nn.Linear(W, 9)

        self.alpha_linear = nn.Linear(W, 1)

        self.rgb_linear = nn.Linear(W, 3)


    def forward(self, x):
        # x [bs*64, 90]
        # input_pts [bs*64, 63]
        # input_views [bs*64,27]
        gpu_tracker = MemTracker()
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts

        gpu_tracker.track()
        for i, l in enumerate(self.pts_linears):
            
            h = F.relu(self.pts_linears[i](h))
            
            gpu_tracker.track()
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        alpha = F.softplus(alpha + 0.5)

        roughness = self.rough_linear(h)
        roughness = F.softplus(roughness - 1.)
        bottle_neck = self.bottle_neck_linear(h)
        [normal, diffuse_rgb, spec_tint] = self.norm_color_tint_linear(h).split((3, 3, 3), dim = -1)

        normal = normal / (normal.norm(dim = -1, keepdim=True) + 1e-7)
        dot_res = torch.sum(normal * input_views, dim = -1, keepdim=True)
        reflect_res = 2. * dot_res * normal - input_views
        ide_res = self.ide(reflect_res, roughness)

        h = torch.cat([bottle_neck, ide_res, dot_res], dim = -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

            if i in self.skips:
                all_inputs = torch.cat([bottle_neck, ide_res, dot_res], dim = -1)
                h = torch.cat([all_inputs, h], -1)

        spec_rgb = self.rgb_linear(h)
        spec_rgb = F.sigmoid(spec_rgb) * F.sigmoid(spec_tint)
        diffuse_rgb = torch.sigmoid(diffuse_rgb - log(3.))
    
        rgb = linear_to_srgb(spec_rgb + diffuse_rgb)
        
        outputs = torch.cat([rgb, alpha, normal], -1)

        return outputs
    
    @staticmethod
    def get_grad(func_val: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:  # remember: grad goes from low to high
        grad, = torch.autograd.grad(func_val, inputs, 
            torch.ones_like(func_val, device = func_val.device), retain_graph = True
        )
        grad_norm = grad.norm(dim = -1, keepdim = True)
        return grad / torch.maximum(torch.full_like(grad_norm, 1e-5), grad_norm)

class WeightedNormalLoss(nn.Module):
    def __init__(self, size_average = False):
        super().__init__()
        self.size_average = size_average        # average (per point, not per ray)
    
    # weight (ray_num, point_num)
    def forward(self, weight:torch.Tensor, d_norm: torch.Tensor, p_norm: torch.Tensor) -> torch.Tensor:
        dot_diff = 1. - torch.sum(d_norm * p_norm, dim = -1)
        return torch.mean(weight * dot_diff) if self.size_average == True else torch.sum(weight * dot_diff)

class BackFaceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, weight:torch.Tensor, normal: torch.Tensor, ray_d: torch.Tensor) -> torch.Tensor:
        return torch.mean(weight * F.relu(torch.sum(normal * ray_d, dim = -1)))