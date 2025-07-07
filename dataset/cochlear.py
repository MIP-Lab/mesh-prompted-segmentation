from torch.utils.data import Dataset
import glob
import os
import nibabel as nib
import numpy as np
from monai.transforms import RandRotate, GaussianSmooth, RandGaussianSmooth
from scipy.spatial.transform import Rotation
import time
import torch

class CochlearCTCrop(Dataset):

    def __init__(self, data_root, cases, aug=False, crop_size=(128, 128, 128), fixed_structure=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.cases = cases
        self.aug = aug
        self.crop_size = crop_size
        self.fixed_structure = fixed_structure
    
    def __getitem__(self, index):

        case = self.cases[index]
        pre = nib.load(f'{self.data_root}/Pre_img/{case}.nii.gz').get_fdata().astype(np.float32)
        pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{case}.pkl', allow_pickle=True)
        ac_vtx = pat_vtx[: 2852, :]
        st_vtx = pat_vtx[2852: 6196, :]
        sv_vtx = pat_vtx[6196: 9328, :]
        cc_vtx = pat_vtx[9328: , :]
        # md_mask = nib.load(f'{self.data_root}/MD_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # st_mask = nib.load(f'{self.data_root}/ST_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # sv_mask = nib.load(f'{self.data_root}/SV_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # cc_mask = nib.load(f'{self.data_root}/CC_mask/{case}.nii.gz').get_fdata().astype(np.float32)

        another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
        another_case = self.cases[another_index]

        another_pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{another_case}.pkl', allow_pickle=True)
        another_ac_vtx = another_pat_vtx[: 2852, :]
        another_st_vtx = another_pat_vtx[2852: 6196, :]
        another_sv_vtx = another_pat_vtx[6196: 9328, :]
        another_cc_vtx = another_pat_vtx[9328: , :]
        # another_md_mask = nib.load(f'{self.data_root}/MD_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_st_mask = nib.load(f'{self.data_root}/ST_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_sv_mask = nib.load(f'{self.data_root}/SV_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_cc_mask = nib.load(f'{self.data_root}/CC_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)

        another_md_sdf = nib.load(f'{self.data_root}/SDF/MD/{another_case}.nii.gz').get_fdata().astype(np.float32)
        another_st_sdf = nib.load(f'{self.data_root}/SDF/ST/{another_case}.nii.gz').get_fdata().astype(np.float32)
        another_sv_sdf = nib.load(f'{self.data_root}/SDF/SV/{another_case}.nii.gz').get_fdata().astype(np.float32)
        another_cc_sdf = nib.load(f'{self.data_root}/SDF/CC/{another_case}.nii.gz').get_fdata().astype(np.float32)

        if self.aug:
            # [pre, md_mask, st_mask, sv_mask, cc_mask], vtx = self.do_augmentation([pre, md_mask, st_mask, sv_mask, cc_mask], vtx)
            [pre], pat_vtx = self.do_augmentation([pre], pat_vtx)
            ac_vtx = pat_vtx[: 2852, :]
            st_vtx = pat_vtx[2852: 6196, :]
            sv_vtx = pat_vtx[6196: 9328, :]
            cc_vtx = pat_vtx[9328: , :]
        
        rand_structure_id = np.random.permutation([0, 1, 2, 3])[0]
        if self.fixed_structure is not None:
            rand_structure_id = self.fixed_structure

        if rand_structure_id == 0:
            res = {
                'pre': pre[None], 'vtx': ac_vtx, 'another_pat_vtx': another_ac_vtx, 'another_pat_sdf': another_md_sdf[None]
            }
        elif rand_structure_id == 1:
            res = {
                'pre': pre[None], 'vtx': st_vtx, 'another_pat_vtx': another_st_vtx, 'another_pat_sdf': another_st_sdf[None]
            }
        elif rand_structure_id == 2:
            res = {
                'pre': pre[None], 'vtx': sv_vtx, 'another_pat_vtx': another_sv_vtx, 'another_pat_sdf': another_sv_sdf[None]
            }
        else:
            res = {
                'pre': pre[None], 'vtx': cc_vtx, 'another_pat_vtx': another_cc_vtx, 'another_pat_sdf': another_cc_sdf[None]
            }
        
        res['index'] = index
        res['another_index'] = another_index
        res['structure'] = rand_structure_id

        return res

    def check_data(self, index):
        from pycip.chart import Chart, ChartGroup
        import matplotlib.pyplot as plt

        cg = ChartGroup(1, 2)
        c1 = cg.get_chart(1, 1)
        c2 = cg.get_chart(1, 2)
        data = self.__getitem__(index)

        c1.slice(data['pre'][0], orientation='z', slice_index=64)
        pp1 = data['vtx'][data['vtx'][:, 2].round(0) == 64]
        c2.slice(data['another_pat_sdf'][0], orientation='z', slice_index=64)
        pp2 = data['another_pat_vtx'][data['another_pat_vtx'][:, 2].round(0) == 64]

        c1.scatter(pp1[:, 0], pp1[:, 1], color='red')
        c2.scatter(pp2[:, 0], pp2[:, 1], color='red')

        c1.title(f'index={data["index"]}. structure={data["structure"]}')
        c2.title(f'index={data["another_index"]}.')
        
        
        plt.show()

    def load_atlas(self):
        atlas_nii = nib.load('%s/atlas.nii.gz' % self.data_root)
        atlas, atlas_affine = atlas_nii.get_fdata().astype(np.float32), atlas_nii.affine
        atlas = 2 * (atlas - atlas.min()) / (atlas.max() - atlas.min()) - 1
        atlas_vtx = np.load(open('%s/atlas_vxt.pkl' % self.data_root, 'rb'), allow_pickle=True)
        atlas_cc_mask = nib.load('%s/atlas_CC.nii.gz' % self.data_root).get_fdata().astype(np.float32)
        atlas_stsv_mask = nib.load('%s/atlas_STSV.nii.gz' % self.data_root).get_fdata().astype(np.float32)
        return atlas, atlas_affine, atlas_vtx, atlas_cc_mask, atlas_stsv_mask

    def do_augmentation(self, images, points):
        theta = np.pi * 15 * (2 * np.random.uniform(0, 1) - 1) / 180 # -15 to 15 degrees
        rot_axis = ['x', 'y', 'z'][np.random.randint(0, 3)]
        params = {'x': 0, 'y': 0, 'z': 0}
        params[rot_axis] = (theta, theta)
        rot_matrix = Rotation.from_euler(rot_axis, theta).as_matrix()
        
        images = [image[None, :, :, :] for image in images]

        # 4D input (c * H*W*D)
        T = RandRotate(range_x=params['x'], range_y=params['y'], range_z=params['z'], prob=1, padding_mode='zeros')
        images = [T(image) for image in images]
        
        # blur images
        # sigma = 0
        # if np.random.uniform(0, 1) > 0.25:
        #     sigma = np.random.uniform(0, 1) + 0.5 # 0.5 to 1.5
        #     S = RandGaussianSmooth(sigma_x=(sigma, sigma), sigma_y=(sigma, sigma), sigma_z=(sigma, sigma), prob=1)
        #     images = [S(image) for image in images]

        center = self.crop_size[0] / 2 - 1
        points -= center
        points = points @ rot_matrix
        points += center

        images = [image[0].numpy() for image in images]
        # print(theta, sigma, rot_axis)
        return images, points


    def __len__(self):
        return len(self.cases)
        


class CochlearCTCropEff(Dataset):

    def __init__(self, data_root, cases, aug=False, crop_size=(128, 128, 128), fixed_structure=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.cases = cases
        self.aug = aug
        self.crop_size = crop_size
        self.fixed_structure = fixed_structure
    
    def __getitem__(self, index):

        case = self.cases[index]
        pre = nib.load(f'{self.data_root}/Pre_img/{case}.nii.gz').get_fdata().astype(np.float32)
        pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{case}.pkl', allow_pickle=True)
        ac_vtx = pat_vtx[: 2852, :]
        st_vtx = pat_vtx[2852: 6196, :]
        sv_vtx = pat_vtx[6196: 9328, :]
        cc_vtx = pat_vtx[9328: , :]
        
        # md_mask = nib.load(f'{self.data_root}/MD_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # st_mask = nib.load(f'{self.data_root}/ST_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # sv_mask = nib.load(f'{self.data_root}/SV_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # cc_mask = nib.load(f'{self.data_root}/CC_mask/{case}.nii.gz').get_fdata().astype(np.float32)

        another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
        another_case = self.cases[another_index]
        
        another_pre = nib.load(f'{self.data_root}/Pre_img/{another_case}.nii.gz').get_fdata().astype(np.float32)
        another_pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{another_case}.pkl', allow_pickle=True)
        another_ac_vtx = another_pat_vtx[: 2852, :]
        another_st_vtx = another_pat_vtx[2852: 6196, :]
        another_sv_vtx = another_pat_vtx[6196: 9328, :]
        another_cc_vtx = another_pat_vtx[9328: , :]
        
        rand_structure_id = np.random.permutation([0, 1, 2, 3])[0]
        if self.fixed_structure is not None:
            rand_structure_id = self.fixed_structure
        if rand_structure_id == 0:
            vtx = ac_vtx
            mask = nib.load(f'{self.data_root}/MD_mask/{case}.nii.gz').get_fdata().astype(np.float32)
            another_vtx = another_ac_vtx
            another_mask = nib.load(f'{self.data_root}/MD_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
            another_sdf = nib.load(f'{self.data_root}/SDF/MD/{another_case}.nii.gz').get_fdata().astype(np.float32)
        elif rand_structure_id == 1:
            vtx = st_vtx
            mask = nib.load(f'{self.data_root}/ST_mask/{case}.nii.gz').get_fdata().astype(np.float32)
            another_vtx = another_st_vtx
            another_mask = nib.load(f'{self.data_root}/ST_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
            another_sdf = nib.load(f'{self.data_root}/SDF/ST/{another_case}.nii.gz').get_fdata().astype(np.float32)
        elif rand_structure_id == 2:
            vtx = sv_vtx
            mask = nib.load(f'{self.data_root}/SV_mask/{case}.nii.gz').get_fdata().astype(np.float32)
            another_vtx = another_sv_vtx
            another_mask = nib.load(f'{self.data_root}/SV_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
            another_sdf = nib.load(f'{self.data_root}/SDF/SV/{another_case}.nii.gz').get_fdata().astype(np.float32)
        elif rand_structure_id == 3:
            vtx = cc_vtx
            mask = nib.load(f'{self.data_root}/CC_mask/{case}_CC.nii.gz').get_fdata().astype(np.float32)
            another_vtx = another_cc_vtx
            another_mask = nib.load(f'{self.data_root}/CC_mask/{another_case}_CC.nii.gz').get_fdata().astype(np.float32)
            another_sdf = nib.load(f'{self.data_root}/SDF/CC/{another_case}.nii.gz').get_fdata().astype(np.float32)

        if self.aug:
            # [pre, md_mask, st_mask, sv_mask, cc_mask], vtx = self.do_augmentation([pre, md_mask, st_mask, sv_mask, cc_mask], vtx)
            [pre, mask], vtx = self.do_augmentation([pre, mask], vtx)
        
        res = {
                'pre': pre[None], 'vtx': vtx, 'mask': mask[None], 'another_pat_mask': another_mask[None],
                'another_pat_vtx': another_vtx, 'another_pat_sdf': another_sdf[None], 'another_pat_pre': another_pre[None]
            }

        res['index'] = index
        res['another_index'] = another_index
        res['structure'] = rand_structure_id

        return res

    def check_data(self, index):
        from pycip.chart import Chart, ChartGroup
        import matplotlib.pyplot as plt

        cg = ChartGroup(1, 2)
        c1 = cg.get_chart(1, 1)
        c2 = cg.get_chart(1, 2)
        data = self.__getitem__(index)

        c1.slice(data['pre'][0], orientation='z', slice_index=64)
        pp1 = data['vtx'][data['vtx'][:, 2].round(0) == 64]
        c2.slice(data['another_pat_sdf'][0], orientation='z', slice_index=64)
        pp2 = data['another_pat_vtx'][data['another_pat_vtx'][:, 2].round(0) == 64]

        c1.scatter(pp1[:, 0], pp1[:, 1], color='red')
        c2.scatter(pp2[:, 0], pp2[:, 1], color='red')

        c1.title(f'index={data["index"]}. structure={data["structure"]}')
        c2.title(f'index={data["another_index"]}.')
        
        
        plt.show()

    def load_atlas(self):
        atlas_nii = nib.load('%s/atlas.nii.gz' % self.data_root)
        atlas, atlas_affine = atlas_nii.get_fdata().astype(np.float32), atlas_nii.affine
        atlas = 2 * (atlas - atlas.min()) / (atlas.max() - atlas.min()) - 1
        atlas_vtx = np.load(open('%s/atlas_vxt.pkl' % self.data_root, 'rb'), allow_pickle=True)
        atlas_cc_mask = nib.load('%s/atlas_CC.nii.gz' % self.data_root).get_fdata().astype(np.float32)
        atlas_stsv_mask = nib.load('%s/atlas_STSV.nii.gz' % self.data_root).get_fdata().astype(np.float32)
        return atlas, atlas_affine, atlas_vtx, atlas_cc_mask, atlas_stsv_mask

    def do_augmentation(self, images, points):
        theta = np.pi * 15 * (2 * np.random.uniform(0, 1) - 1) / 180 # -15 to 15 degrees
        rot_axis = ['x', 'y', 'z'][np.random.randint(0, 3)]
        params = {'x': 0, 'y': 0, 'z': 0}
        params[rot_axis] = (theta, theta)
        rot_matrix = Rotation.from_euler(rot_axis, theta).as_matrix()
        
        images = [image[None, :, :, :] for image in images]

        # 4D input (c * H*W*D)
        T = RandRotate(range_x=params['x'], range_y=params['y'], range_z=params['z'], prob=1, padding_mode='zeros')
        images = [T(image) for image in images]
        
        # blur images
        # sigma = 0
        # if np.random.uniform(0, 1) > 0.25:
        #     sigma = np.random.uniform(0, 1) + 0.5 # 0.5 to 1.5
        #     S = RandGaussianSmooth(sigma_x=(sigma, sigma), sigma_y=(sigma, sigma), sigma_z=(sigma, sigma), prob=1)
        #     images = [S(image) for image in images]

        center = self.crop_size[0] / 2 - 1
        points -= center
        points = points @ rot_matrix
        points += center

        images = [image[0].numpy() for image in images]
        # print(theta, sigma, rot_axis)
        return images, points


    def __len__(self):
        return len(self.cases)
        
    
class CochlearCTCropTest(Dataset):

    def __init__(self, data_root, cases, aug=False, crop_size=(128, 128, 128), fixed_structure=None) -> None:
        super().__init__()
        self.data_root = data_root
        self.cases = cases
        self.aug = aug
        self.crop_size = crop_size
        self.fixed_structure = fixed_structure
    
    def __getitem__(self, index):

        case = self.cases[index]
        pre = nib.load(f'{self.data_root}/Pre_img/{case}.nii.gz').get_fdata().astype(np.float32)
        pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{case}.pkl', allow_pickle=True)
        ac_vtx = pat_vtx[: 2852, :]
        st_vtx = pat_vtx[2852: 6196, :]
        sv_vtx = pat_vtx[6196: 9328, :]
        cc_vtx = pat_vtx[9328: , :]
        # md_mask = nib.load(f'{self.data_root}/MD_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # st_mask = nib.load(f'{self.data_root}/ST_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # sv_mask = nib.load(f'{self.data_root}/SV_mask/{case}.nii.gz').get_fdata().astype(np.float32)
        # cc_mask = nib.load(f'{self.data_root}/CC_mask/{case}.nii.gz').get_fdata().astype(np.float32)

        another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
        another_case = self.cases[another_index]

        # another_pat_vtx = np.load(f'{self.data_root}/PrePost_vtx/{another_case}.pkl', allow_pickle=True)
        another_pat_vtx = np.load(f'{self.data_root}/../atlas_vxt.pkl', allow_pickle=True)
        another_ac_vtx = another_pat_vtx[: 2852, :]
        another_st_vtx = another_pat_vtx[2852: 6196, :]
        another_sv_vtx = another_pat_vtx[6196: 9328, :]
        another_cc_vtx = another_pat_vtx[9328: , :]
        # another_md_mask = nib.load(f'{self.data_root}/MD_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_st_mask = nib.load(f'{self.data_root}/ST_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_sv_mask = nib.load(f'{self.data_root}/SV_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)
        # another_cc_mask = nib.load(f'{self.data_root}/CC_mask/{another_case}.nii.gz').get_fdata().astype(np.float32)

        another_md_sdf = nib.load(f'{self.data_root}/../atlas_SDF/md_sdf.nii.gz').get_fdata().astype(np.float32)
        another_st_sdf = nib.load(f'{self.data_root}/../atlas_SDF/st_sdf.nii.gz').get_fdata().astype(np.float32)
        another_sv_sdf = nib.load(f'{self.data_root}/../atlas_SDF/sv_sdf.nii.gz').get_fdata().astype(np.float32)
        another_cc_sdf = nib.load(f'{self.data_root}/../atlas_SDF/cc_sdf.nii.gz').get_fdata().astype(np.float32)
        
        rand_structure_id = np.random.permutation([0, 1, 2, 3])[0]
        if self.fixed_structure is not None:
            rand_structure_id = self.fixed_structure

        res = {
                'pre': pre[None], 
                'ac_vtx': ac_vtx, 'another_ac_vtx': another_ac_vtx, 'another_ac_sdf': another_md_sdf[None],
                'st_vtx': st_vtx, 'another_st_vtx': another_st_vtx, 'another_st_sdf': another_st_sdf[None],
                'sv_vtx': sv_vtx, 'another_sv_vtx': another_sv_vtx, 'another_sv_sdf': another_sv_sdf[None],
                'cc_vtx': cc_vtx, 'another_cc_vtx': another_cc_vtx, 'another_cc_sdf': another_cc_sdf[None],
            }
        
        res['index'] = index
        res['another_index'] = another_index

        return res

    def __len__(self):
        return len(self.cases)

if __name__ == '__main__':

    import json

    DATA_ROOT = 'E:/dingjie/mipresearch/data/cochlear/data_128_inAtlas_99clip'

    dataset = json.load(open('../dataset.json'))
    train_ds = CochlearCTCropEff(DATA_ROOT + '/train', cases=dataset['train'], aug=True)

    train_ds.check_data(0)