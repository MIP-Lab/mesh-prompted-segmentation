from tkinter import N
import torch
from torch.nn.functional import relu
import numpy as np
# from torch.nn.functional import leaky_relu
from monai.networks.blocks.warp import Warp
# from voxelmorph.torch.modelio import LoadableModel, store_config_args

# def relu(x):
#     return leaky_relu(x, negative_slope=0.2)

class GlobalNet(torch.nn.Module):

    def __init__(self, grid_size, batch_size=1, init_transform=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.batch_size = batch_size
        init_transform = np.array([init_transform] * batch_size)
        self.init_transform = torch.nn.Parameter(torch.tensor(init_transform, dtype=torch.float32), requires_grad=True)
        grid = torch.stack(torch.meshgrid(
            torch.tensor([i for i in range(grid_size[0])]),
            torch.tensor([j for j in range(grid_size[1])]),
            torch.tensor([k for k in range(grid_size[2])]),
        indexing='ij'), axis=0)
        self.register_buffer('grid', grid)
        # down path 1
        # conv block 1
        self.dp1_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=8, kernel_size=7, stride=1, padding='same')
        self.dp1_bn1 = torch.nn.BatchNorm3d(num_features=8)
        # conv block 2
        self.dp1_conv2 = torch.nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.dp1_bn2 = torch.nn.BatchNorm3d(num_features=8)
        # conv block 3
        self.dp1_conv3 = torch.nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same')
        self.dp1_bn3 = torch.nn.BatchNorm3d(num_features=8)
        self.dp1_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 2
        # conv block 1
        self.dp2_conv1 = torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.dp2_bn1 = torch.nn.BatchNorm3d(num_features=16)
        # conv block 2
        self.dp2_conv2 = torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.dp2_bn2 = torch.nn.BatchNorm3d(num_features=16)
        # conv block 3
        self.dp2_conv3 = torch.nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='same')
        self.dp2_bn3 = torch.nn.BatchNorm3d(num_features=16)
        self.dp2_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 3
        # conv block 1
        self.dp3_conv1 = torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp3_bn1 = torch.nn.BatchNorm3d(num_features=32)
        # conv block 2
        self.dp3_conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp3_bn2 = torch.nn.BatchNorm3d(num_features=32)
        # conv block 3
        self.dp3_conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp3_bn3 = torch.nn.BatchNorm3d(num_features=32)
        self.dp3_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottom_conv = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.bottom_bn = torch.nn.BatchNorm3d(num_features=64)

        # fully connected
        self.linear = torch.nn.Linear(in_features=self.batch_size*64*16*16*16, out_features=12)
    
    def forward(self, x):
        
        # down path 1
        x = self.dp1_conv1(x)
        x = self.dp1_bn1(x)
        x = relu(x)
        x_prime = x
        x = self.dp1_conv2(x)
        x = self.dp1_bn2(x)
        x = relu(x)
        x = self.dp1_conv3(x)
        x = self.dp1_bn3(x)
        x = x + x_prime
        x = relu(x)
        x = self.dp1_maxpool(x)

        # down path 2
        x = self.dp2_conv1(x)
        x = self.dp2_bn1(x)
        x = relu(x)
        x_prime = x
        x = self.dp2_conv2(x)
        x = self.dp2_bn2(x)
        x = relu(x)
        x = self.dp2_conv3(x)
        x = self.dp2_bn3(x)
        x = x + x_prime
        x = relu(x)
        x = self.dp2_maxpool(x)

        # down path 3
        x = self.dp3_conv1(x)
        x = self.dp3_bn1(x)
        x = relu(x)
        x_prime = x
        x = self.dp3_conv2(x)
        x = self.dp3_bn2(x)
        x = relu(x)
        x = self.dp3_conv3(x)
        x = self.dp3_bn3(x)
        x = x + x_prime
        x = relu(x)
        x = self.dp3_maxpool(x)

        # bottleneck
        x = self.bottom_conv(x)
        x = self.bottom_bn(x)
        x = relu(x)
        
        # fully connected
        x_flatten = torch.flatten(x)
        theta = self.linear(x_flatten).reshape((self.batch_size, -1))
        theta += self.init_transform

        warped_grid = self.warp_grid(self.grid, theta)
        ddf = warped_grid - self.grid
        # ddf *= 0.0001
        return ddf
    
    def warp_grid(self, grid, theta):
        # grid=grid_reference
        num_batch = self.batch_size
        theta = theta.reshape((-1, 3, 4))
        print(theta)
        size = grid.shape[1: ]
        grid = torch.concat([grid.reshape((-1, 3)).T, torch.ones([1, size[0]*size[1]*size[2]], device=grid.device)], axis=0)
        grid = torch.reshape(torch.tile(torch.reshape(grid, [-1]), [num_batch]), [num_batch, 4, -1])
        grid_warped = torch.matmul(theta, grid)
        return torch.reshape(torch.transpose(grid_warped, 1, 2), [num_batch, 3, size[0], size[1], size[2]])


class LocalNetNoBN(torch.nn.Module):

    def __init__(self, grid_size, batch_size=1) -> None:
        super(LocalNetNoBN, self).__init__()
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.ddf_bias = torch.nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float32), requires_grad=True)
        # down path 1
        # conv block 1
        self.dp1_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding='same')
        # conv block 2
        self.dp1_conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        # conv block 3
        self.dp1_conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp1_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 2
        # conv block 1
        self.dp2_conv1 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        # conv block 2
        self.dp2_conv2 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        # conv block 3
        self.dp2_conv3 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.dp2_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 3
        # conv block 1
        self.dp3_conv1 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        # conv block 2
        self.dp3_conv2 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        # conv block 3
        self.dp3_conv3 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.dp3_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottom_conv = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        # self.bottom_skip_conv = torch.nn.Conv3d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 3
        self.up3_deconv1 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up3_conv1 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.up3_conv2 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        # self.up3_skip_conv = torch.nn.Conv3d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 2
        self.up2_deconv1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up2_conv1 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.up2_conv2 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        # self.up2_skip_conv = torch.nn.Conv3d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 1
        self.up1_deconv1 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up1_conv1 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up1_conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up1_skip_conv = torch.nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding='same')

        # self.encoded = []
        # self.decoded = []
    
    

    def forward(self, x):
        
        # down path 1
        x = self.dp1_conv1(x)
        x = relu(x)
        x_prime1 = x
        x = self.dp1_conv2(x)
        x = relu(x)
        x = self.dp1_conv3(x)
        x = x + x_prime1
        x = relu(x)
        x = self.dp1_maxpool(x)

        # down path 2
        x = self.dp2_conv1(x)
        x = relu(x)
        x_prime2 = x
        x = self.dp2_conv2(x)
        x = relu(x)
        x = self.dp2_conv3(x)
        x = x + x_prime2
        x = relu(x)
        x = self.dp2_maxpool(x)

        # down path 3
        x = self.dp3_conv1(x)
        x = relu(x)
        x_prime3 = x
        x = self.dp3_conv2(x)
        x = relu(x)
        x = self.dp3_conv3(x)
        x = x + x_prime3
        x = relu(x)
        x = self.dp3_maxpool(x)

        # bottleneck
        x = self.bottom_conv(x)
        x = relu(x)

        # up path 3
        x_deconv = self.up3_deconv1(x)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime3
        x1 = self.up3_conv1(x1)
        x1 = relu(x1)
        x1 = self.up3_conv2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # up path 2
        x_deconv = self.up2_deconv1(x)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime2
        x1 = self.up2_conv1(x1)
        x1 = relu(x1)
        x1 = self.up2_conv2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # up path 1
        x_deconv = self.up1_deconv1(x)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime1
        x1 = self.up1_conv1(x1)
        x1 = relu(x1)
        x1 = self.up1_conv2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # final layer
        ddf = self.up1_skip_conv(x) + self.ddf_bias.reshape([1, 3, 1, 1, 1])

        return ddf
    
    def additive_upsampling(self, x):
        size = [x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2]
        reshaped = torch.nn.functional.interpolate(x, size, mode='trilinear')
        splited = torch.split(reshaped, x.shape[1] // 2, dim=1)
        stacked = torch.stack(splited, dim=5)
        summed = stacked.sum(dim=5)
        return summed

class GlabolLocalNet(torch.nn.Module):

    def __init__(self, grid_size, batch_size=1) -> None:
        super().__init__()
        self.grid_size = grid_size
        [X, Y, Z] = np.meshgrid(range(grid_size[0]), range(grid_size[1]), range(grid_size[2]))
        grid = torch.from_numpy(np.stack([Y, X, Z], axis=0)).type(torch.float32)
        self.register_buffer('grid', grid)
        self.batch_size = batch_size
        self.global_net = GlobalNet(grid_size, batch_size)
        self.local_net = LocalNet(grid_size, batch_size)
        self.warp_layer = Warp(mode='bilinear', padding_mode='zeros')

    def forward(self, x):

        moving = x[:, :1, :]
        fixed = x[:, 1:, :]

        linear_warped_grid = self.global_net(x)
        linear_ddf = linear_warped_grid - self.grid
        linear_warped_moving = self.warp_layer(moving, linear_ddf)

        new_x = torch.cat([linear_warped_moving, fixed], dim=1)
        nonlinear_ddf = self.local_net(new_x)

        return linear_ddf, nonlinear_ddf, linear_warped_moving  


class LocalNet(torch.nn.Module):

    def __init__(self, grid_size, batch_size=1) -> None:
        super(LocalNet, self).__init__()
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.ddf_bias = torch.nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float32), requires_grad=True)
        # down path 1
        # conv block 1
        self.dp1_conv1 = torch.nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp1_bn1 = torch.nn.BatchNorm3d(num_features=32)
        # conv block 2
        self.dp1_conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp1_bn2 = torch.nn.BatchNorm3d(num_features=32)
        # conv block 3
        self.dp1_conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.dp1_bn3 = torch.nn.BatchNorm3d(num_features=32)
        self.dp1_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 2
        # conv block 1
        self.dp2_conv1 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.dp2_bn1 = torch.nn.BatchNorm3d(num_features=64)
        # conv block 2
        self.dp2_conv2 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.dp2_bn2 = torch.nn.BatchNorm3d(num_features=64)
        # conv block 3
        self.dp2_conv3 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.dp2_bn3 = torch.nn.BatchNorm3d(num_features=64)
        self.dp2_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # down path 3
        # conv block 1
        self.dp3_conv1 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.dp3_bn1 = torch.nn.BatchNorm3d(num_features=128)
        # conv block 2
        self.dp3_conv2 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.dp3_bn2 = torch.nn.BatchNorm3d(num_features=128)
        # conv block 3
        self.dp3_conv3 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.dp3_bn3 = torch.nn.BatchNorm3d(num_features=128)
        self.dp3_maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        # bottleneck
        self.bottom_conv = torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same')
        self.bottom_bn = torch.nn.BatchNorm3d(num_features=256)
        # self.bottom_skip_conv = torch.nn.Conv3d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 3
        self.up3_deconv1 = torch.nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up3_deconvbn = torch.nn.BatchNorm3d(num_features=128)
        self.up3_conv1 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.up3_bn1 = torch.nn.BatchNorm3d(num_features=128)
        self.up3_conv2 = torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.up3_bn2 = torch.nn.BatchNorm3d(num_features=128)
        # self.up3_skip_conv = torch.nn.Conv3d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 2
        self.up2_deconv1 = torch.nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up2_deconvbn = torch.nn.BatchNorm3d(num_features=64)
        self.up2_conv1 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.up2_bn1 = torch.nn.BatchNorm3d(num_features=64)
        self.up2_conv2 = torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.up2_bn2 = torch.nn.BatchNorm3d(num_features=64)
        # self.up2_skip_conv = torch.nn.Conv3d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding='same')

        # up path 1
        self.up1_deconv1 = torch.nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up1_deconvbn = torch.nn.BatchNorm3d(num_features=32)
        self.up1_conv1 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up1_bn1 = torch.nn.BatchNorm3d(num_features=32)
        self.up1_conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.up1_bn2 = torch.nn.BatchNorm3d(num_features=32)
        self.up1_skip_conv = torch.nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding='same')

        # self.encoded = []
        # self.decoded = []
    
    

    def forward(self, x):
        
        # down path 1
        x = self.dp1_conv1(x)
        x = self.dp1_bn1(x)
        x = relu(x)
        x_prime1 = x
        x = self.dp1_conv2(x)
        x = self.dp1_bn2(x)
        x = relu(x)
        x = self.dp1_conv3(x)
        x = self.dp1_bn3(x)
        x = x + x_prime1
        x = relu(x)
        x = self.dp1_maxpool(x)

        # down path 2
        x = self.dp2_conv1(x)
        x = self.dp2_bn1(x)
        x = relu(x)
        x_prime2 = x
        x = self.dp2_conv2(x)
        x = self.dp2_bn2(x)
        x = relu(x)
        x = self.dp2_conv3(x)
        x = self.dp2_bn3(x)
        x = x + x_prime2
        x = relu(x)
        x = self.dp2_maxpool(x)

        # down path 3
        x = self.dp3_conv1(x)
        x = self.dp3_bn1(x)
        x = relu(x)
        x_prime3 = x
        x = self.dp3_conv2(x)
        x = self.dp3_bn2(x)
        x = relu(x)
        x = self.dp3_conv3(x)
        x = self.dp3_bn3(x)
        x = x + x_prime3
        x = relu(x)
        x = self.dp3_maxpool(x)

        # bottleneck
        x = self.bottom_conv(x)
        x = self.bottom_bn(x)
        x = relu(x)

        # up path 3
        x_deconv = self.up3_deconv1(x)
        x_deconv = self.up3_deconvbn(x_deconv)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime3
        x1 = self.up3_conv1(x1)
        x1 = self.up3_bn1(x1)
        x1 = relu(x1)
        x1 = self.up3_conv2(x1)
        x1 = self.up3_bn2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # up path 2
        x_deconv = self.up2_deconv1(x)
        x_deconv = self.up2_deconvbn(x_deconv)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime2
        x1 = self.up2_conv1(x1)
        x1 = self.up2_bn1(x1)
        x1 = relu(x1)
        x1 = self.up2_conv2(x1)
        x1 = self.up2_bn2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # up path 1
        x_deconv = self.up1_deconv1(x)
        x_deconv = self.up1_deconvbn(x_deconv)
        x_deconv = relu(x_deconv)
        x_addup = self.additive_upsampling(x)
        x1 = x_deconv + x_addup
        x1_prime = x1 + x_prime1
        x1 = self.up1_conv1(x1)
        x1 = self.up1_bn1(x1)
        x1 = relu(x1)
        x1 = self.up1_conv2(x1)
        x1 = self.up1_bn2(x1)
        x = x1 + x1_prime
        x = relu(x)

        # final layer
        ddf = self.up1_skip_conv(x) + self.ddf_bias.reshape([1, 3, 1, 1, 1])

        return ddf
    
    def additive_upsampling(self, x):
        size = [x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2]
        reshaped = torch.nn.functional.interpolate(x, size, mode='trilinear')
        splited = torch.split(reshaped, x.shape[1] // 2, dim=1)
        stacked = torch.stack(splited, dim=5)
        summed = stacked.sum(dim=5)
        return summed

if __name__ == '__main__':

    device = torch.device('cuda')

    np.random.seed(0)
    sz = 96
    patchsize = [sz, sz, sz]
    x1 = np.random.random(patchsize).astype(np.float32)
    x2 = np.random.random(patchsize).astype(np.float32)
    x = np.array([x1, x2])
    x = x[None, :]
    x = torch.from_numpy(x)

    net = GlabolLocalNet(patchsize)
    net.to(device)
    x = x.to(device)
    y = net(x)

    print(1)

