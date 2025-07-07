from torch.utils.data import Dataset
import glob
import os
import nibabel as nib
import numpy as np
from monai.transforms import RandRotate, GaussianSmooth, RandGaussianSmooth
from scipy.spatial.transform import Rotation
import time
import torch
import pyvista as pv
from scipy.ndimage import distance_transform_edt

class CommonCrop(Dataset):

    def __init__(self, data_root, cases, structure_list, aug=False, crop_size=(128, 128, 128)) -> None:
        super().__init__()
        self.data_root = data_root
        self.cases = cases
        self.aug = aug
        self.crop_size = crop_size
        self.structure_list = structure_list
    
    def __getitem__(self, index):

        case = self.cases[index]

        vols = nib.load(f'{self.data_root}/{case}/merged_vol.nii.gz')
        vols_data, affine = vols.get_fdata().astype(np.float32), vols.affine

        img = vols_data[0]
        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-5)) - 1

        vols_data[0] = img

        res = {
            'index': index,
            'vols': vols_data,
            'voxsz': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
        }

        verts = np.load(f'{self.data_root}/{case}/merged_sampled_verts.npz')
        for k in self.structure_list:
            res[k + '_verts'] = verts[k]

        return res

    def check_data(self, index, save_name=None):
        from pycip.chart import Chart, ChartGroup
        import matplotlib.pyplot as plt

        data = self.__getitem__(index)

        cg = ChartGroup(2, len(self.structure_list) + 1)
        c0 = cg.get_chart(1, 1)
        c0.slice(data['vols'][0], orientation='y', slice_index=64)

        for i in range(len(self.structure_list)):
            c1 = cg.get_chart(1, i + 2)
            c2 = cg.get_chart(2, i + 2)

            verts = data[self.structure_list[i] + '_verts']
            center = verts.mean(axis=0).round().astype(np.int32)

            c1.slice(data['vols'][i * 2 + 1], orientation='y', slice_index=center[1])
            c2.slice(data['vols'][i * 2 + 2], orientation='y', slice_index=center[1])

            pp1 = verts[verts[:, 1].round(0) == center[1]]
            c1.scatter(pp1[:, 0], pp1[:, 2], color='red')
            c2.scatter(pp1[:, 0], pp1[:, 2], color='red')

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def __len__(self):
        return len(self.cases)
    
class CommonCropBatch(Dataset):

    def __init__(self, data_root, split,
                 cases, structure_list, aug=False, crop_size=(128, 128, 128)) -> None:
        self.data_root = data_root
        self.cases = cases
        self.crop_size = crop_size
        self.structure_list = structure_list
        self.images = np.load(f'{data_root}/batch/{split}_images.npz')
        self.masks = np.load(f'{data_root}/batch/{split}_masks.npz')
        self.sdfs = np.load(f'{data_root}/batch/{split}_sdfs.npz')
        self.verts = np.load(f'{data_root}/batch/{split}_verts.npz', allow_pickle=True)
    
    def __getitem__(self, index):

        case = self.cases[index]

        affine = nib.load(f'{self.data_root}/processed/{case}/combined_mask.nii.gz').affine

        img = self.images[case]
        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-5)) - 1
        res = {
            'index': index,
            'img': img.astype(np.float32), 
            'sdfs': self.sdfs[case].astype(np.float32),
            'masks': self.masks[case].astype(np.float32),
            'voxsz': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
        }

        for k in self.structure_list:
            res[k + '_verts'] = self.verts[case].item()[k]

        return res
    
    def check_data(self, index, save_name=None):
        from pycip.chart import Chart, ChartGroup
        import matplotlib.pyplot as plt

        data = self.__getitem__(index)

        cg = ChartGroup(2, len(self.structure_list) + 1)
        c0 = cg.get_chart(1, 1)
        c0.slice(data['img'], orientation='y', slice_index=64)

        for i in range(len(self.structure_list)):
            c1 = cg.get_chart(1, i + 2)
            c2 = cg.get_chart(2, i + 2)

            verts = data[self.structure_list[i] + '_verts']
            center = verts.mean(axis=0).round().astype(np.int32)

            c1.slice(data['sdfs'][i], orientation='y', slice_index=center[1])
            c2.slice(data['masks'][i], orientation='y', slice_index=center[1])

            pp1 = verts[verts[:, 1].round(0) == center[1]]
            c1.scatter(pp1[:, 0], pp1[:, 2], color='red')
            c2.scatter(pp1[:, 0], pp1[:, 2], color='red')

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def __len__(self):
        return len(self.cases)

def loc_sdf(verts1, sdf1, mask1, verts2, loc_r, crop_size, loc_r_fixed=None):
    center1 = verts1.mean(axis=0)
    center2 = verts2.mean(axis=0)

    bbox_min1 = verts1.min(axis=0)
    bbox_max1 = verts2.max(axis=0)

    if loc_r_fixed is None:
        r = loc_r
    else:
        # no random loc
        r = 0
    if np.abs(center1 - center2).max() <= r:
        return verts1.astype(np.float32), sdf1.astype(np.float32), mask1.astype(np.float32)

    if r > 0:
        rand_shift = np.random.randint(0, r, 3)
        #rand_shift = np.zeros(3)
    else:
        rand_shift = np.zeros(3)
    rand_center2 = center2 + rand_shift

    shift = rand_center2 - center1
    for i in range(3):
        if shift[i] + bbox_min1[i] < 0:
            shift[i] = -bbox_min1[i]
        if shift[i] + bbox_max1[i] > crop_size[i]:
            shift[i] = crop_size[i] - bbox_max1[i]
    
    if loc_r_fixed is not None:
        
        # add additional fixed shift
        new_bbox_min1 = bbox_min1 + shift
        new_bbox_max1 = bbox_max1 + shift

        fixed_shift = np.zeros(3)
        xyz_shift = [[], [], []]
        for i in range(3):
            if new_bbox_min1[i] - loc_r_fixed >= 0:
                xyz_shift[i].append(-loc_r_fixed)
            if new_bbox_max1[i] + loc_r_fixed < crop_size[i]:
                xyz_shift[i].append(loc_r_fixed)
        for i in range(3):
            if len(xyz_shift[i]) == 0:
                xyz_shift[i].append(0)
        assert len(xyz_shift[0]) > 1 or len(xyz_shift[1]) > 1 or len(xyz_shift[2]) > 0
        fixed_shift[0] = np.random.choice(xyz_shift[0], 1)[0]
        fixed_shift[1] = np.random.choice(xyz_shift[1], 1)[0]
        fixed_shift[2] = np.random.choice(xyz_shift[2], 1)[0]
        shift += fixed_shift
    
    bg_mask = np.zeros_like(mask1)
    new_sdf = np.zeros_like(sdf1)

    new_center = np.array(crop_size) / 2 + shift
    new_center = new_center.round().astype(np.int32)

    hs = 64
    x1 = min(hs, new_center[0])
    x2 = min(hs, hs * 2 - new_center[0])
    y1 = min(hs, new_center[1])
    y2 = min(hs, hs * 2 - new_center[1])
    z1 = min(hs, new_center[2])
    z2 = min(hs, hs * 2 - new_center[2])

    bg_mask[
        new_center[0] - x1: new_center[0] + x2,
        new_center[1] - y1: new_center[1] + y2,
        new_center[2] - z1: new_center[2] + z2,
    ] = mask1[
        hs - x1: hs + x2,
        hs - y1: hs + y2,
        hs - z1: hs + z2]

    new_sdf[
        new_center[0] - x1: new_center[0] + x2,
        new_center[1] - y1: new_center[1] + y2,
        new_center[2] - z1: new_center[2] + z2,
    ] = sdf1[
        hs - x1: hs + x2,
        hs - y1: hs + y2,
        hs - z1: hs + z2]

    new_sdf_ex = distance_transform_edt(1 - bg_mask).astype(np.float32)
    new_sdf_ex = (new_sdf_ex - new_sdf_ex.min()) / (new_sdf_ex.max() - new_sdf_ex.min())

    new_sdf[bg_mask == 0] = new_sdf_ex[bg_mask == 0] * (-1)

    new_verts = (verts1 + shift).astype(np.float32)

    return new_verts.astype(np.float32), new_sdf.astype(np.float32), bg_mask.astype(np.float32)

class CommonCropBatchTwoSampleOneStructure(Dataset):

    def __init__(self, data_root, split,
                 cases, structure_list, aug=False, crop_size=(128, 128, 128), fixed_structure=None, loc_sdf=False, loc_r=3) -> None:
        self.data_root = data_root
        self.cases = cases
        self.crop_size = crop_size
        self.structure_list = structure_list
        self.images = np.load(f'{data_root}/batch/{split}_images.npz')
        self.masks = np.load(f'{data_root}/batch/{split}_masks.npz')
        self.sdfs = np.load(f'{data_root}/batch/{split}_sdfs.npz')
        verts = np.load(f'{data_root}/batch/{split}_verts.npz', allow_pickle=True)
        self.verts = {}
        for case in cases:
            case_verts = {}
            for k, v in verts[case].item().items():
                case_verts[k] = v
            self.verts[case] = case_verts
        self.fixed_structure = fixed_structure
        self.loc_sdf = loc_sdf
        self.loc_r = loc_r
    
    def __getitem__(self, index):

        rand_structure_id = np.random.choice([i for i in range(len(self.structure_list))], 1)[0]
        if self.fixed_structure is not None:
            rand_structure_id = self.fixed_structure

        case = self.cases[index]
        # case = 's0686'

        affine = nib.load(f'{self.data_root}/processed/{case}/combined_mask.nii.gz').affine

        img = self.images[case]
        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-5)) - 1
        res = {
            'index': index,
            'img1': img.astype(np.float32), 
            'sdf1': self.sdfs[case][rand_structure_id].astype(np.float32),
            'mask1': self.masks[case][rand_structure_id].astype(np.float32),
            'verts1': self.verts[case][self.structure_list[rand_structure_id]].astype(np.float32),
        }

        another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
        another_case = self.cases[another_index]

        another_img = self.images[another_case]
        another_img = 2 * ((another_img - another_img.min()) / (another_img.max() - another_img.min() + 1e-5)) - 1

        res.update({
            'index2': another_index,
            'img2': another_img.astype(np.float32), 
            'sdf2': self.sdfs[another_case][rand_structure_id].astype(np.float32),
            'mask2': self.masks[another_case][rand_structure_id].astype(np.float32),
            'verts2': self.verts[another_case][self.structure_list[rand_structure_id]].astype(np.float32)
        })

        if self.loc_sdf:
            
            verts1_loc, sdf1_loc, mask1_loc = loc_sdf(res['verts1'], res['sdf1'], 
                                                      res['mask1'], res['verts2'], self.loc_r, self.crop_size)
            res['verts1_loc'] = verts1_loc
            res['sdf1_loc'] = sdf1_loc
            res['mask1_loc'] = mask1_loc

        return res

    def __len__(self):
        return len(self.cases)

class CommonCropBatchTwoSampleAllStructures(Dataset):

    def __init__(self, data_root, split,
                 cases, structure_list, aug=False, crop_size=(128, 128, 128), fixed_structure=None, 
                 fixed_case2=None) -> None:
        self.data_root = data_root
        self.cases = cases
        self.crop_size = crop_size
        self.structure_list = structure_list
        self.images = np.load(f'{data_root}/batch/{split}_images.npz')
        # self.masks = np.load(f'{data_root}/batch/{split}_masks.npz')
        # self.sdfs = np.load(f'{data_root}/batch/{split}_sdfs.npz')
        # self.verts = np.load(f'{data_root}/batch/{split}_verts.npz', allow_pickle=True)
        self.fixed_structure = fixed_structure
        self.fixed_case2 = fixed_case2
    
    def __getitem__(self, index):

        case = self.cases[index]

        affine = nib.load(f'{self.data_root}/processed/{case}/combined_mask.nii.gz').affine

        img = self.images[case]
        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-5)) - 1
        mask1 = nib.load(f'{self.data_root}/processed/{case}/combined_mask.nii.gz').get_fdata().astype(np.float32)

        res = {
            'index': index,
            'img1': img.astype(np.float32), 
            'mask1': mask1,
            'voxsz1': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]]),
            # 'verts1': self.verts[case].item()
        }

        if self.fixed_case2 is None:
            another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
            another_case = self.cases[another_index]
        else:
            another_index = -1
            another_case = self.fixed_case2

        another_img = self.images[another_case]
        another_img = 2 * ((another_img - another_img.min()) / (another_img.max() - another_img.min() + 1e-5)) - 1
        mask2 = nib.load(f'{self.data_root}/processed/{another_case}/combined_mask.nii.gz').get_fdata().astype(np.float32)

        res.update({
            'index2': another_index,
            'img2': another_img.astype(np.float32), 
            'mask2': mask2,
            'voxsz2': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]]),
            # 'verts2': self.verts[another_case].item()
        })

        return res

    def __len__(self):
        return len(self.cases)
    
class CommonCropBatchTwoSampleAllStructuresTest(Dataset):

    def __init__(self, data_root, split,
                 cases, structure_list, aug=False, crop_size=(128, 128, 128), 
                 fixed_structure=None, fixed_case2=None, im_key='img', loc_sdf=False, loc_r=3, loc_r_fixed=None) -> None:
        self.data_root = data_root
        self.cases = cases
        self.crop_size = crop_size
        self.structure_list = structure_list
        self.fixed_structure = fixed_structure
        self.fixed_case2 = fixed_case2
        self.im_key = im_key
        self.loc_sdf = loc_sdf
        self.loc_r = loc_r
        self.loc_r_fixed = loc_r_fixed
    
    def __getitem__(self, index):

        case = self.cases[index]

        affine = nib.load(f'{self.data_root}/processed/{case}/{self.im_key}.nii.gz').affine

        img = nib.load(f'{self.data_root}/processed/{case}/{self.im_key}.nii.gz').get_fdata().astype(np.float32)
        img = 2 * ((img - img.min()) / (img.max() - img.min() + 1e-5)) - 1

        mask1 = np.zeros([len(self.structure_list)] + list(self.crop_size), dtype=np.float32)

        for i, k in enumerate(self.structure_list):
            try:
                cur_mask = nib.load(f'{self.data_root}/processed/{case}/{k}_mask.nii.gz').get_fdata().astype(np.float32)
            except FileNotFoundError:
                cur_mask = nib.load(f'{self.data_root}/processed/{case}/{k}.nii.gz').get_fdata().astype(np.float32)
            mask1[i] = cur_mask

        res = {
            'index': index,
            'img1': img.astype(np.float32), 
            'mask1': mask1,
            'voxsz1': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]]),
        }

        if self.fixed_case2 is None:
            another_index = np.random.choice([i for i in range(len(self.cases)) if i != index], 1)[0]
            another_case = self.cases[another_index]
        else:
            another_index = -1
            another_case = self.fixed_case2

        another_img = nib.load(f'{self.data_root}/processed/{another_case}/{self.im_key}.nii.gz').get_fdata().astype(np.float32)
        another_img = 2 * ((another_img - another_img.min()) / (another_img.max() - another_img.min() + 1e-5)) - 1
        
        mask2 = np.zeros([len(self.structure_list)] + list(self.crop_size), dtype=np.float32)
        for i, k in enumerate(self.structure_list):
            try:
                cur_mask = nib.load(f'{self.data_root}/processed/{another_case}/{k}_mask.nii.gz').get_fdata().astype(np.float32)
            except FileNotFoundError:
                cur_mask = nib.load(f'{self.data_root}/processed/{another_case}/{k}.nii.gz').get_fdata().astype(np.float32)
            mask2[i] = cur_mask

        res.update({
            'index2': another_index,
            'img2': another_img.astype(np.float32), 
            'mask2': mask2,
            'voxsz2': np.abs([affine[0, 0], affine[1, 1], affine[2, 2]]),
            # 'verts2': self.verts[another_case].item()
        })

        for i, k in enumerate(self.structure_list):
            mesh1 = pv.read(f'{self.data_root}/processed/{case}/{k}_mesh.ply')
            mesh2 = pv.read(f'{self.data_root}/processed/{another_case}/{k}_mesh.ply')
            verts1 = np.array(mesh1.points).astype(np.float32)
            tris1 = np.array(mesh1.faces.reshape((-1, 4))[:, 1:]).astype(np.int32)
            verts2 = np.array(mesh2.points).astype(np.float32)
            tris2 = np.array(mesh2.faces.reshape((-1, 4))[:, 1:]).astype(np.int32)
            res[f'{k}_verts1'] = verts1
            res[f'{k}_tris1'] = tris1
            res[f'{k}_verts2'] = verts2
            res[f'{k}_tris2'] = tris2

            sdf1 = nib.load(f'{self.data_root}/processed/{case}/{k}_sdf.nii.gz').get_fdata().astype(np.float32)
            sdf2 = nib.load(f'{self.data_root}/processed/{another_case}/{k}_sdf.nii.gz').get_fdata().astype(np.float32)

            res[f'{k}_sdf1'] = sdf1
            res[f'{k}_sdf2'] = sdf2

            if self.loc_sdf:
                verts2_loc, sdf2_loc, mask2_loc = loc_sdf(verts2, sdf2, res['mask2'][i], verts1, self.loc_r, self.crop_size, self.loc_r_fixed)
                res[f'{k}_verts2_loc'] = verts2_loc
                res[f'{k}_sdf2_loc'] = sdf2_loc
                res[f'{k}_mask2_loc'] = mask2_loc
            else:
                res[f'{k}_verts2_loc'] = verts2
                res[f'{k}_sdf2_loc'] = sdf2
                res[f'{k}_mask2_loc'] = mask2

        return res

    def __len__(self):
        return len(self.cases)

if __name__ == '__main__':

    import json
    import time
    import pycip

    cases = os.listdir('G:/mipresearch/mps_data/body/processed')

    # body_ds = CommonCrop('G:/mipresearch/mps_data/body/processed', cases, ['heart', 'liver', 'kidney_left', 'kidney_right', 'pancreas'])
    # body_ds.check_data(0)
    # for i in range(len(body_ds)):
    #     body_ds.check_data(i, save_name=f'../check_data/body/{body_ds.cases[i]}.jpg')

    # dataset = json.load(open('G:/mipresearch/mps_data/cochlear/cochlear_dataset.json'))
    # cases = dataset['train']
    # cochlear_ds = CommonCropBatch('G:/mipresearch/mps_data/cochlear', split='train', cases=cases, structure_list=['md', 'st', 'sv', 'cc'])
    # cochlear_ds.check_data(100)

    # dataset = json.load(open('G:/mipresearch/mps_data/body/body_dataset.json'))
    # cases = dataset['val']
    # body_ds = CommonCropBatch('G:/mipresearch/mps_data/body', split='val', 
    #                           cases=cases, structure_list=['heart', 'liver', 'kidney_left', 'kidney_right', 'pancreas'])
    # body_ds.check_data(0)

    dataset = json.load(open('G:/mipresearch/mps_data/body/body_dataset.json'))
    cases = dataset['val']

    # ds1 = CommonCropBatchTwoSampleAllStructuresTest('G:/mipresearch/mps_data/body', 'val', cases,
    #                                             structure_list=['heart', 'liver', 'kidney_left', 'kidney_right', 'pancreas']
    #                                             )
    ds2 = CommonCropBatchTwoSampleAllStructuresTest('G:/mipresearch/mps_data/body', 'val', cases,
                                               structure_list=['heart']
                                               , loc_sdf=True)
    
    datum = ds2[0]

    nib.save(nib.Nifti1Image(datum['heart_sdf2'], np.eye(4)), 'T/sdf2.nii.gz')
    # nib.save(nib.Nifti1Image(datum['mask1'], np.eye(4)), 'T/mask1.nii.gz')
    nib.save(nib.Nifti1Image(pycip.api.mesh2contour(datum['heart_verts2'], (128, 128, 128)), np.eye(4)), 'T/verts2.nii.gz')
    nib.save(nib.Nifti1Image(datum['heart_sdf2_loc'], np.eye(4)), 'T/sdf2_loc.nii.gz')
    # nib.save(nib.Nifti1Image(datum['mask1_loc'], np.eye(4)), 'T/mask1_loc.nii.gz')
    nib.save(nib.Nifti1Image(pycip.api.mesh2contour(datum['heart_verts2_loc'], (128, 128, 128)), np.eye(4)), 'T/verts2_loc.nii.gz')

    # pl = pv.Plotter()
    # pl.add_points(datum['verts1'], color='red')
    # pl.add_points(datum['verts1_loc'], color='green')
    # pl.add_points(datum['verts2'], color='yellow')

    # pl.show()

    # t1 = time.time()
    # for i in range(5):
    #     a = ds1[i]
    
    # t2 = time.time()

    # for i in range(5):
    #     a = ds2[i]
    
    # t3 = time.time()

    # print(t2 - t1, t3 - t2)

