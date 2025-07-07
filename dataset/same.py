from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from skimage.filters import gaussian
import torchio as tio
import torch
import pyvista as pv

class SAMECommonTest(Dataset):

    def __init__(self, data_root, cases, atlas_case, structure_list,
                 std_spacing=(1, 1, 1), blur_sigma=0, img_key='img'
                 ):
        self.data_root = data_root
        self.structure_list = structure_list
        self.std_spacing = np.array(std_spacing)
        self.cases = cases
        self.blur_sigma = blur_sigma
        self.img_key = img_key
        self.atlas = self._get_data(atlas_case)
        
    def __len__(self):
        return len(self.cases)

    def _get_data(self, case):
        # change orientation to LPS, consistent with HeasSAME
        img = nib.load(f'{self.data_root}/processed/{case}/{self.img_key}.nii.gz')

        img_data = np.array(img.dataobj).astype(np.float32)
        img_data = np.flip(img_data, 0)
        img_data = np.flip(img_data, 1)
        # Treat isotropic
        new_affine = np.eye(4)

        img_data = 2 * ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-5)) - 1

        data_resampled_255 = 255 * ((img_data + 1) / 2)
        data_resampled_255 -= 50

        if self.blur_sigma != 0:
            data_resampled_255 = gaussian(data_resampled_255, sigma=self.blur_sigma)

        res = {'img': data_resampled_255}

        for k in self.structure_list:
            mesh = pv.read(f'{self.data_root}/processed/{case}/{k}_mesh.ply')
            verts = np.array(mesh.points).astype(np.float32)
            verts[:, 0] = img.shape[0] - verts[:, 0]
            verts[:, 1] = img.shape[1] - verts[:, 1]
            res[f'{k}_verts'] = verts

        return res

    def __getitem__(self, index):

        case_data = self._get_data(self.cases[index])

        for k, v in self.atlas.items():
            case_data[f'atlas_{k}'] = v.copy()
        
        return case_data
    
if __name__ == '__main__':

    import json
    import pycip

    DATA_ROOT = 'G:/mipresearch/mps_data'

    dataset = json.load(open(f'{DATA_ROOT}/body/body_dataset.json'))

    #structures = ['md', 'st', 'sv', 'cc']
    structures = ['heart', 'liver', 'kidney_left', 'kidney_right', 'pancreas']


    test_dataset = SAMECommonTest(f'{DATA_ROOT}/body', cases=dataset['test'], atlas_case='s0686', structure_list=structures)

    datum = test_dataset[0]

    nib.save(nib.Nifti1Image(datum['img'], np.eye(4)), 'T/temp1.nii.gz')

    contour = pycip.api.mesh2contour_all([datum[f'{k}_verts'] for k in structures], shape=(128, 128, 128))

    nib.save(nib.Nifti1Image(contour, np.eye(4)), 'T/temp2.nii.gz')