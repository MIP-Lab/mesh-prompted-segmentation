import os
from env import DATA_ROOT

# from dataset.cochlear import CochlearCTCrop
from dataset.common import CommonCrop, CommonCropBatch, CommonCropBatchTwoSampleAllStructuresTest, CommonCropBatchTwoSampleOneStructure
from pycip.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils.transform import SpatialTransformer, trilinear_interpolation3_torch
import random
from pytorch3d.loss import (
    chamfer_distance,
)
#from pytorch3d.loss.chamfer import hyperbolic_chamfer_distance
import json
from model.vfa.vfa import VFA
import numpy as np
import nibabel as nib
from model.vfa.utils import grid_sampler
from pytorch3d.loss.chamfer import hyperbolic_chamfer_distance, chamfer_distance
import pyvista as pv
import pycip

params = {
    "dataset": "cochlear",
    "model":{
        "in_channels": 1,
        "in_shape": [128, 128, 128],
        "name":"VFA",
        "skip":0,
        "initialize":0.1,
        "downsamples":4,
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
    
    def test_loss(self, model, input_data):

        res = {'case': test_ds.cases[int(input_data['index'][0].detach().cpu().numpy())]}

        for i, k in enumerate(structures):
            
            x = {
            'f_img': input_data[f'{k}_sdf2_loc'][0][None, None],
            'm_img': input_data['img1'][0][None, None],
            'f_keypoints': input_data[f'{k}_verts2_loc'][0][None]
            }
            y = model(x)

            pat_verts = input_data[f'{k}_verts1'][0].detach().cpu().numpy()
            pat_tris = input_data[f'{k}_tris1'][0].detach().cpu().numpy()
            atlas_verts_loc = input_data[f'{k}_verts2_loc'][0].detach().cpu().numpy()
            atlas_verts = input_data[f'{k}_verts2'][0].detach().cpu().numpy()
            atlas_tris = input_data[f'{k}_tris2'][0].detach().cpu().numpy()

            loss_chamfer, _ = chamfer_distance(y['w_keypoints'], input_data[f'{k}_verts1'][0][None])

            atlas2pat_verts = y['w_keypoints'][0].detach().cpu().numpy()

            atlas2pat_mask = pycip.api.mesh2mask(atlas2pat_verts, atlas_tris, [128, 128, 128], [1, 1, 1])

            pat_mask = (input_data['mask1'][0, i]).float().detach().cpu().numpy()
            atlas_mask = (input_data['mask2'][0, i]).float().detach().cpu().numpy()

            dice = 2 * (atlas2pat_mask * pat_mask).sum() / ((atlas2pat_mask + pat_mask).sum() + 1e-5)
            dice_init = 2 * (atlas_mask * pat_mask).sum() / ((atlas_mask + pat_mask).sum() + 1e-5)
            center1 = input_data[f'{k}_verts1'][0].mean(axis=0)
            center2 = input_data[f'{k}_verts2'][0].mean(axis=0)
            centerdist = torch.sqrt(((center1 - center2) ** 2).sum())

            # pl = pv.Plotter()
            # pl.add_points(pat_verts, color='red')
            # pl.add_points(atlas_verts_loc, color='green')
            # pl.add_points(atlas_verts, color='blue')
            # pl.add_points(atlas2pat_verts, color='yellow')

            # pl.show()

            res[f'{k}_dice'] = dice
            res[f'{k}_dice_init'] = dice_init
            res[f'{k}_centerdist'] = centerdist
            res[f'{k}_chamfer'] = loss_chamfer

        return {'grid': y['grid']}, res

random.seed(2025)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

inshape = [128, 128, 128]  
model = VFA(params=params, device='cuda' ,encoder2=False)

dataset = json.load(open(f'data_split/{params["dataset"]}_dataset.json'))

structures = ['md', 'st', 'sv', 'cc']
# structures = ['kidney_right']


test_ds = CommonCropBatchTwoSampleAllStructuresTest(f'{DATA_ROOT}/cochlear',  split='test', im_key='pre',
                                                cases=dataset['test'], structure_list=structures, 
                                                fixed_case2='atlas', loc_sdf=True, loc_r=3)


test_loader = DataLoader(dataset=test_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='exp_midl', name='Exp_cochlear-vfa-sdf-chamfer-loc_r5_downsample4_1')

trainer.evaluate(model, exp_id=1, epoch='best', test_loader=test_loader, device='cuda', tag='loc_3')

print(1)



