import os
from env import DATA_ROOT

# from dataset.cochlear import CochlearCTCrop
from dataset.common import CommonCrop, CommonCropBatch, CommonCropBatchTwoSampleOneStructure
from pycip.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils.transform import SpatialTransformer, trilinear_interpolation3_torch
import random
from pytorch3d.loss import (
    chamfer_distance,
)
from pytorch3d.loss.chamfer import hyperbolic_chamfer_distance
import json
from model.vfa.vfa import VFA
import numpy as np
import nibabel as nib

params = {
    "dataset": "cochlear",
    "model":{
        "in_channels": 1,
        "in_shape": [128, 128, 128],
        "name":"VFA",
        "skip":0,
        "initialize":0.1,
        "downsamples":2,
        "start_channels":4,
        "matching_channels":4,
        "int_steps":0,
        "affine":0
    },
    "loss":{
	"train":{
		"MSE":{
		    "weight":1.0
		},
        "Dice": {
            'weight': 1.0
        },
		"Grad":{
		    "weight":1.0
		},
	},
    }
}

class MyTrainer(Trainer):

    def train_loss(self, model, input_data):

        f_img = input_data['sdf1_loc'][0]
        m_img = input_data['img2'][0]
        
        f_seg = input_data['mask1_loc'][0]
        m_seg = input_data['mask2'][0]

        f_verts = input_data['verts1_loc'][0]
        m_verts = input_data['verts2'][0]

        x = {
            'f_img': f_img[None, None],
            'm_img': m_img[None, None],
            'f_seg': f_seg[None, None],
            'm_seg': m_seg[None, None],
            'f_keypoints': f_verts[None]
            
        }
        y = model(x)

        loss_reg = model.calc_grad_loss(y, phase='train')
        loss_dice = model.calc_dice_loss(y, x, labels=[1], phase='train')
        loss_chamfer, _ = hyperbolic_chamfer_distance(y['w_keypoints'], m_verts[None])

        loss = 1.0 * loss_reg + 0.0 * loss_dice + 1.0 * loss_chamfer

        return {'grid': y['grid']}, {'reg': loss_reg, 'dice': loss_dice, 'chamfer': loss_chamfer,
                                     'total_loss': loss}

random.seed(2025)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

inshape = [128, 128, 128]  
model = VFA(params=params, device='cuda')

dataset = json.load(open(f'data_split/{params["dataset"]}_dataset.json'))

structures = ['md', 'st', 'sv', 'cc']
#structures = ['heart', 'liver', 'kidney_left', 'kidney_right', 'pancreas']


train_ds = CommonCropBatchTwoSampleOneStructure(f'{DATA_ROOT}/{params["dataset"]}',  split='train', cases=dataset['train'], structure_list=structures, loc_sdf=True, loc_r=5)
val_ds = CommonCropBatchTwoSampleOneStructure(f'{DATA_ROOT}/{params["dataset"]}', split='val', cases=dataset['val'], structure_list=structures, loc_sdf=True, loc_r=5)
train_loader = DataLoader(dataset=train_ds, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='exp_new', name='cochlear-vfa-sdf-chamfer-loc_r5_downsample2')

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

trainer.fit_and_val(model, optimizer, train_loader=train_loader, val_loader=val_loader, total_iterations=100000,
                    log_per_iteration=10, save_per_iteration=750, save_best=True)

print(1)



