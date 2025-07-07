import numpy as np
import os
from subprocess import check_call, DEVNULL, STDOUT, PIPE
import time
from .cdb_parser.cdb_parser import CDBParser
from numpy.lib.ufunclike import fix
import math
import re
try:
    import nibabel as nib
except:
    pass
try:
    from .utils import cur_folder
    from . import utils
except ImportError:
    from utils import cur_folder
    import utils

def image_transform(
    in_data,
    in2out_matrix,
    in_shape,
    in_spacing,
    in_direction,
    out_shape,
    out_spacing,
    out_direction,
    interpolation='trilinear',
    remove_temp_folder=True,
    dtype=np.int16,
):  
    
    in_dtype = in_data.dtype
    out_dtype = dtype
    # dtype_code = {
    #     np.uint8: 1,
    #     np.int16: 2,
    #     np.int32: 4,
    #     np.float32: 6,
    #     np.float64: 7
    # }

    dtype_code = {
        'uint8': 1,
        'int16': 2,
        'int32': 4,
        'float32': 6,
        'float64': 7
    }

    interpolation_code = {
        'trilinear': 1,
        'partial_volume': 2,
        'nearest': 0
    }

    folder_id = utils.create_temp_folder()

    in_image_path = f'{cur_folder}/temp/{folder_id}/input.im'
    out_image_path = f'{cur_folder}/temp/{folder_id}/out.im'
    in_transformation_path = f'{cur_folder}/temp/{folder_id}/input.transf'
    out_transformation_path = f'{cur_folder}/temp/{folder_id}/out.transf'
    config_path = f'{cur_folder}/temp/{folder_id}/amir.in'

    out2in_mat = np.linalg.inv(in2out_matrix)

    # prepare transformation file
    with open(f'{cur_folder}/template/amir_affine.transf') as f:
        template = f.read()
        transf = template.format(
            source_shape=utils.vec2str(in_shape, 'int'),
            target_shape=utils.vec2str(out_shape, 'int'),
            source_spacing=utils.vec2str(in_spacing),
            target_spacing=utils.vec2str(out_spacing),
            source_direction=utils.vec2str(in_direction, 'int'),
            target_direction=utils.vec2str(out_direction, 'int'),
            affine_matrix=utils.mat2str(out2in_mat)
        )
        with open(in_transformation_path, 'w') as f:
            f.write(transf)
    
    # prepare amir input file
    with open(f'{cur_folder}/template/amir_apply_affine.in') as f:
        template = f.read()
        config = template.format(
            source_image=in_image_path,
            transformed_image=out_image_path,
            input_transformation=in_transformation_path,
            output_transformation=out_transformation_path,
            source_shape=utils.vec2str(in_shape, 'int'),
            target_shape=utils.vec2str(out_shape, 'int'),
            source_spacing=utils.vec2str(in_spacing),
            target_spacing=utils.vec2str(out_spacing),
            source_direction=utils.vec2str(in_direction, 'int'),
            target_direction=utils.vec2str(out_direction, 'int'),
            source_dtype=dtype_code[str(in_dtype)],
            target_dtype=dtype_code[str(out_dtype)],
            interpolation=interpolation_code[interpolation]
        )
        with open(config_path, 'w') as f:
            f.write(config)
    
    in_data_im = utils.volume2im(in_data)
    in_data_im.tofile(in_image_path)

    # cmd = f'{cur_folder}/bin/amir.exe {config_path}'
    # print(cmd)
    # os.system(cmd)
    check_call([f'{cur_folder}/bin/amir.exe', config_path], stdout=DEVNULL, stderr=STDOUT)
    
    out_data = utils.im2volume(out_image_path, shape=out_shape, dtype=out_dtype)

    if remove_temp_folder:
        utils.delete_temp_folder(folder_id)

    return out_data

def change_image_direction(volume, from_direction, to_direction):
    from_direction_abs = np.abs(from_direction).tolist()
    to_direction_abs = np.abs(to_direction).tolist()
    cx = from_direction_abs.index(to_direction_abs[0])
    cy = from_direction_abs.index(to_direction_abs[1])
    cz = from_direction_abs.index(to_direction_abs[2])

    volume = np.transpose(volume, [cx, cy, cz])

    if from_direction[cx] * to_direction[0] < 0:
        volume = volume[::-1, :, :]
    if from_direction[cy] * to_direction[1] < 0:
        volume = volume[:, ::-1, :]
    if from_direction[cz] * to_direction[2] < 0:
        volume = volume[:, :, ::-1]
    
    return volume

def change_point_direction(points, from_direction, to_direction, out_shape):
    from_direction_abs = np.abs(from_direction).tolist()
    to_direction_abs = np.abs(to_direction).tolist()
    cx = from_direction_abs.index(to_direction_abs[0])
    cy = from_direction_abs.index(to_direction_abs[1])
    cz = from_direction_abs.index(to_direction_abs[2])

    points = points[:, [cx, cy, cz]]

    if from_direction[cx] * to_direction[0] < 0:
        points[:, 0] = out_shape[0] - 1 - points[:, 0]
    if from_direction[cy] * to_direction[1] < 0:
        points[:, 1] = out_shape[1] - 1 - points[:, 1]
    if from_direction[cz] * to_direction[2] < 0:
        points[:, 2] = out_shape[2] - 1 - points[:, 2]
    
    return points

# def mesh2contour(
#     vertices_list,
#     out_shape,
#     sampling_factor=0.5
#     ):
#     voxel = np.zeros(out_shape, dtype=np.int16)

#     for label, vertices in enumerate(vertices_list):
#         rnd_indice = np.random.choice(vertices.shape[0], size=int(sampling_factor * vertices.shape[0]), replace=False)
#         vertices = vertices[rnd_indice]
#         l = 0
#         for x, y, z in vertices:
#             x = int(round(x))
#             y = int(round(y))
#             z = int(round(z))

#             for i in range(-l, l + 1):
#                 for j in range(-l, l + 1):
#                     for k in range(-l, l + 1):
#                         if x+i < 0 or x+i >= out_shape[0] or y+j < 0 or y+j >= out_shape[1] or z+k < 0 or z+k >= out_shape[2]: continue
#                         voxel[x+i][y+j][z+k] = label + 1
#     return voxel

def point_transform(
    points,
    in2out_matrix,
    in_shape,
    in_spacing,
    in_direction,
    out_shape,
    out_spacing,
    out_direction,
):
    in2out_mat_vox = world_matrix_to_voxel(
        in2out_matrix, 
        in_shape, in_spacing, in_direction, 
        out_shape, out_spacing, out_direction
    )
    points_homo = np.ones((len(points), 4))
    points_homo[:, : 3] = points
    out_points_homo = (in2out_mat_vox @ points_homo.T).T
    out_points = out_points_homo[:, : 3]

    return out_points

def world_matrix_to_voxel(
    world_matrix,
    in_shape,
    in_spacing,
    in_direction,
    out_shape,
    out_spacing,
    out_direction
):

    P2W_in = np.array([
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 1
    ]).reshape(4, 4).astype(np.float32)

    P2W_out = np.array([
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 1
    ]).reshape(4, 4).astype(np.float32)

    # if direction is [1, 2, -3], the third colume of in_spacing_mat will be negated.
    # if direction is [2, 1, 3], the first colume of in_spacing_mat will be placed in the second column of P2W_in,
    for i, ind in enumerate(in_direction):
        P2W_in[abs(ind) - 1, i] = in_spacing[i] * (1 if ind > 0 else -1)
        
    for i, ind in enumerate(out_direction):
        P2W_out[abs(ind) - 1, i] = out_spacing[i] * (1 if ind > 0 else -1)

    T_in = np.array([
        1, 0, 0, -(in_shape[0] - 1) / 2,
        0, 1, 0, -(in_shape[1] - 1) / 2,
        0, 0, 1, -(in_shape[2] - 1) / 2,
        0, 0, 0, 1
    ]).reshape(4, 4).astype(np.float32)

    T_out = np.array([
        1, 0, 0, -(out_shape[0] - 1) / 2,
        0, 1, 0, -(out_shape[1] - 1) / 2,
        0, 0, 1, -(out_shape[2] - 1) / 2,
        0, 0, 0, 1
    ]).reshape(4, 4).astype(np.float32)

    # first, move input image to its center, then scale by input voxel spacing, then the apply world matrix, 
    # then re-scale by output spacing, finally move from center to the output image corner.
    voxel_matrix = np.linalg.inv(T_out) @ np.linalg.inv(P2W_out) @ world_matrix @ P2W_in @ T_in
    print(P2W_in @ T_in)
    return voxel_matrix

def registration(fixed, moving, type_of_transform, roi_target=[85, 178, 105, 170, 43, 120], region_mask=None, remove_temp_folder=True):

    folder_id = utils.create_temp_folder()
    print(f'using folder {folder_id}')

    config_path = f'{cur_folder}/temp/{folder_id}/amir_registration.in'
    fixed_image_path = f'{cur_folder}/temp/{folder_id}/target.nii'
    moving_image_path = f'{cur_folder}/temp/{folder_id}/source.nii'
    moving2fixed_image_path = f'{cur_folder}/temp/{folder_id}/s_t_affine.nii'
    moving2fixed_transformation_path = f'{cur_folder}/temp/{folder_id}/s_t.transf'
    fixed2moving_image_path = f'{cur_folder}/temp/{folder_id}/t_s_affine.nii'
    fixed2moving_transformation_path = f'{cur_folder}/temp/{folder_id}/t_s.transf'

    if type_of_transform == 'Rigid':
        order = ' '.join(map(str, [3, 2, 1, 4, 6, 5, 0, 0, 0, 0, 0, 0]))
    else:
        order = ' '.join(map(str, [3, 2, 1, 4, 6, 5, 9, 8, 7, 12, 11, 10]))
    
    with open(f'{cur_folder}/template/amir_registration.in') as f:
        template = f.read()
        config = template.format(
            source_image=moving_image_path,
            target_image=fixed_image_path,
            s_t_image=moving2fixed_image_path,
            s_t_transform=moving2fixed_transformation_path,
            t_s_image=fixed2moving_image_path,
            t_s_transform=fixed2moving_transformation_path,
            roi_target='    '.join([str(i) for i in roi_target]),
            order=order
        )
        with open(config_path, 'w') as f:
            f.write(config)
    
    # aba requires fixed and moving has the same datatype. unify them to be float32
    fixed = nib.Nifti1Image(fixed.get_fdata().astype(np.float32), fixed.affine)
    moving = nib.Nifti1Image(moving.get_fdata().astype(np.float32), moving.affine)

    nib.save(fixed, fixed_image_path)
    nib.save(moving, moving_image_path)

    print('doing registration with amir..')

    check_call([f'{cur_folder}/bin/amir.exe', config_path], stdout=DEVNULL, stderr=STDOUT)

    if type_of_transform != 'Deformable':
        s_t_image = nib.load(moving2fixed_image_path)
        # need to create a new image, otherwise will not be able to find the image file after removing temp folder (lazy loading)
        out = nib.Nifti1Image(s_t_image.get_fdata(), s_t_image.affine)
        out.set_data_dtype(fixed.get_data_dtype())
        s_t_transform = parse_affine_transformation_file(moving2fixed_transformation_path)
        if remove_temp_folder:
            utils.delete_temp_folder(folder_id)
        return out, s_t_transform

    aba_config_path = f'{cur_folder}/temp/{folder_id}/aba_registration.in'
    moving_image_path = moving2fixed_image_path
    moving2fixed_image_path = f'{cur_folder}/temp/{folder_id}/s_t_deform.nii'
    fixed2moving_image_path = f'{cur_folder}/temp/{folder_id}/t_s_deform.nii'

    if region_mask is not None:

        mask_nii_path = f'{cur_folder}/temp/{folder_id}/mask.nii'
        mask_binary_path = f'{cur_folder}/temp/{folder_id}/mask.msk'
        nib.save(nib.Nifti1Image(region_mask.get_fdata().astype(np.int8), region_mask.affine), 
        mask_nii_path)

        mask_binary = np.fromfile(mask_nii_path, dtype=np.int8)
        mask_binary = mask_binary[352: ]
        mask_binary.tofile(mask_binary_path)
        mask = f"""
MASK1:
{mask_binary_path}"""
    else:
        mask = ''

    with open(f'{cur_folder}/template/aba_registration.in') as f:
        template = f.read()
        config = template.format(
            source_image=moving_image_path,
            target_image=fixed_image_path,
            boundingbox='85 172 105 162 43 121',
            s_t_image=moving2fixed_image_path,
            t_s_image=fixed2moving_image_path,
            mask=mask
        )
        with open(aba_config_path, 'w') as f:
            f.write(config)

    time.sleep(2)
    print('doing registration with aba..')
    check_call([f'{cur_folder}/bin/aba.exe', aba_config_path], stdout=DEVNULL, stderr=STDOUT)
    
    s_t_image = nib.load(moving2fixed_image_path)
    out = nib.Nifti1Image(s_t_image.get_fdata(), s_t_image.affine)
    out.set_data_dtype(fixed.get_data_dtype())
    if remove_temp_folder:
        utils.delete_temp_folder(folder_id)
    return out, None

def elastix_registration(fixed, moving, moving_points_in_mm=None, remove_temp_folder=True):

    # folder_id = 'temp_1747755954.8353736_65943'
    folder_id = utils.create_temp_folder()
    print(f'using folder {folder_id}')

    elastix_out_folder = f'{cur_folder}/temp/{folder_id}'
    elastix_config_path = f'{cur_folder}/temp/{folder_id}/elastix_registration.txt'
    fixed_image_path = f'{cur_folder}/temp/{folder_id}/target.nii'
    moving_image_path = f'{cur_folder}/temp/{folder_id}/source.nii'
    moving_points_path = f'{cur_folder}/temp/{folder_id}/points.txt'

    fixed = nib.Nifti1Image(fixed.get_fdata().astype(np.float32), fixed.affine)
    moving = nib.Nifti1Image(moving.get_fdata().astype(np.float32), moving.affine)

    nib.save(fixed, fixed_image_path)
    nib.save(moving, moving_image_path)

    with open(f'{cur_folder}/template/elastix_nonrigid.txt') as f:
        template = f.read()
        config = template
        with open(elastix_config_path, 'w') as f:
            f.write(config)
    
    if moving_points_in_mm is not None:
        lines = ['point', str(moving_points_in_mm.shape[0])]
        for i, p in enumerate(moving_points_in_mm):
            lines.append('%.2f %.2f %.2f' % (p[0], p[1], p[2]))
        
        with open(moving_points_path, 'w') as f:
            f.write('\n'.join(lines))
    
    time.sleep(2)
    print('doing registration with elastix..')
    # cmd_register = f'{cur_folder}/bin/elastix.exe -f {fixed_image_path} -m {moving_image_path} -p {elastix_config_path} -out {elastix_out_folder}'
    check_call([f'{cur_folder}/bin/elastix.exe', 
                '-f', fixed_image_path,
                '-m', moving_image_path,
                '-p', elastix_config_path,
                '-out', elastix_out_folder
                ], stdout=DEVNULL, stderr=STDOUT)

    cmd_transform = [
            f'{cur_folder}/bin/transformix.exe',
            '-in', moving_image_path,
            '-out', elastix_out_folder,
            '-tp', f'{elastix_out_folder}/TransformParameters.0.txt'
        ]
    if moving_points_in_mm is not None:
        cmd_transform += [
            '-def', moving_points_path
        ]
    
    check_call(cmd_transform, stdout=DEVNULL, stderr=STDOUT)

    deformed_image = nib.load(f'{elastix_out_folder}/result.nii')
    deformed_image = nib.Nifti1Image(deformed_image.get_fdata(), deformed_image.affine)
    deformed_image.set_data_dtype(fixed.get_data_dtype())

    if moving_points_in_mm is not None:
        transformed_points = []
        with open(f'{elastix_out_folder}/outputpoints.txt') as f:
            lines = f.readlines()
            for line in lines:
                # Regular expression to match the three integers in OutputIndexFixed
                #match = re.search(r'OutputPoint = \[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', line)
                match = re.search(r'OutputPoint = \[\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*\]', line)
                if match:
                    x, y, z = map(float, match.groups())
                    transformed_points.append([x, y, z])

    if remove_temp_folder:
            utils.delete_temp_folder(folder_id)
    
    if moving_points_in_mm is not None:
        return deformed_image, np.array(transformed_points)

    return deformed_image

def aba_registration(fixed, moving, moving_points=None, region_mask=None, bounding_box=None, remove_temp_folder=True):

    folder_id = utils.create_temp_folder()
    print(f'using folder {folder_id}')

    aba_config_path = f'{cur_folder}/temp/{folder_id}/aba_registration.in'
    fixed_image_path = f'{cur_folder}/temp/{folder_id}/target.nii'
    moving_image_path = f'{cur_folder}/temp/{folder_id}/source.nii'
    moving2fixed_deform_path = f'{cur_folder}/temp/{folder_id}/s_t_deform.nii'
    fixed2moving_deform_path = f'{cur_folder}/temp/{folder_id}/t_s_deform.nii'

    # aba requires fixed and moving has the same datatype. unify them to be float32
    fixed = nib.Nifti1Image(fixed.get_fdata().astype(np.float32), fixed.affine)
    moving = nib.Nifti1Image(moving.get_fdata().astype(np.float32), moving.affine)

    nib.save(fixed, fixed_image_path)
    nib.save(moving, moving_image_path)
    
    if region_mask is not None:

        mask_nii_path = f'{cur_folder}/temp/{folder_id}/mask.nii'
        mask_binary_path = f'{cur_folder}/temp/{folder_id}/mask.msk'
        nib.save(nib.Nifti1Image(region_mask.get_fdata().astype(np.int8), region_mask.affine), 
        mask_nii_path)

        mask_binary = np.fromfile(mask_nii_path, dtype=np.int8)
        mask_binary = mask_binary[352: ]
        mask_binary.tofile(mask_binary_path)
        mask = f"""
MASK1:
{mask_binary_path}"""
    else:
        mask = ''

    with open(f'{cur_folder}/template/aba_registration.in') as f:
        template = f.read()
        config = template.format(
            source_image=moving_image_path,
            target_image=fixed_image_path,
            boundingbox=bounding_box or '60 180 20 120 40 180',
            # boundingbox=bounding_box or '60 180 80 200 20 120',
            s_t_image=moving2fixed_deform_path,
            t_s_image=fixed2moving_deform_path,
            mask=mask
        )
        with open(aba_config_path, 'w') as f:
            f.write(config)

    time.sleep(2)
    print('doing registration with aba..')
    check_call([f'{cur_folder}/bin/aba.exe', aba_config_path], stdout=DEVNULL, stderr=STDOUT)
    
    deformed_image = nib.load(moving2fixed_deform_path)
    out = nib.Nifti1Image(deformed_image.get_fdata(), deformed_image.affine)
    out.set_data_dtype(fixed.get_data_dtype())

    if moving_points is not None:
        amir_apply_config_path = f'{cur_folder}/temp/{folder_id}/amir_apply.in'
        input_points_path = f'{cur_folder}/temp/{folder_id}/points_in.txt'
        out_points_path = f'{cur_folder}/temp/{folder_id}/points_out.txt'
        points_txt = []
        for p in moving_points:
            points_txt.append('%.2f %.2f %.2f' % (p[0], p[1], p[2]))
        with open(input_points_path, 'w') as f:
            f.write('\n'.join(points_txt))
        with open(f'{cur_folder}/template/amir_apply_ddf_points.in') as f:
            template = f.read()
            config = template.format(
                source_image=moving_image_path,
                input_points=input_points_path,
                output_points=out_points_path,
                # boundingbox=bounding_box or '60 180 80 200 20 120',
                s_t_ddf=f'{cur_folder}/temp/{folder_id}/s_t_deform.df',
                t_s_ddf=f'{cur_folder}/temp/{folder_id}/t_s_deform.df',
                source_dimension='%d %d %d' % moving.shape
            )
            with open(amir_apply_config_path, 'w') as f:
                f.write(config)
        time.sleep(2)
        print('transforming points..')
        check_call([f'{cur_folder}/bin/amir.exe', amir_apply_config_path], stdout=DEVNULL, stderr=STDOUT)

        with open(out_points_path) as f:
            out_points = []
            out_points_txt = f.readlines()
            for item in out_points_txt:
                t = item.replace('\\n', '')
                t = re.sub("\s\s+" , " ", item)
                t = t.split(' ')
                out_points.append([float(t[1]), float(t[2]), float(t[3])])
            out_points = np.array(out_points)
            
            print(out_points)

    if remove_temp_folder:
        utils.delete_temp_folder(folder_id)

    if moving_points is not None:
        return out, out_points, f'{cur_folder}/temp/{folder_id}/s_t_deform.df', 
    else:
        return out, f'{cur_folder}/temp/{folder_id}/s_t_deform.df'


# currently only support 1 deform transformation
def apply_transforms(fixed, moving, transformlist, remove_temp_folder=True):
    folder_id = utils.create_temp_folder()
    print(f'using folder {folder_id}')

    amir_config_path = f'{cur_folder}/temp/{folder_id}/amir_apply.in'
    moving_image_path = f'{cur_folder}/temp/{folder_id}/source.nii'
    deformed_image_path = f'{cur_folder}/temp/{folder_id}/deformed.nii'

    moving = nib.Nifti1Image(moving.get_fdata().astype(np.float32), moving.affine)
    nib.save(moving, moving_image_path)

    transformFileName = transformlist[0].split('/')[-1]
    transformFilePath = transformlist[0].replace('/', '\\')
    
    tmp_cur_folder = cur_folder.replace('/', '\\')
    print(f'copy {transformFilePath} {tmp_cur_folder}\\temp\{folder_id}')
    os.system(f'copy {transformFilePath} {tmp_cur_folder}\\temp\{folder_id}')
    os.system(f'ren  {tmp_cur_folder}\\temp\{folder_id}\{transformFileName} deformation.df')

    with open(f'{cur_folder}/template/amir_apply_nifti.in') as f:
        template = f.read()
        config = template.format(
            source_image=moving_image_path,
            transformed_image=deformed_image_path,
            input_transformation = f'{cur_folder}/temp/{folder_id}/deformation.df',
            source_dimension=' '.join(map(lambda x: str(int(x)), moving.shape)),
        )
        with open(amir_config_path, 'w') as f:
            f.write(config)

    print('applying transformation with amir..')
    check_call([f'{cur_folder}/bin/amir.exe', amir_config_path], stdout=DEVNULL, stderr=STDOUT)
    deformed_image = nib.load(deformed_image_path)
    if remove_temp_folder:
        utils.delete_temp_folder(folder_id)
    return deformed_image

def parse_affine_transformation_file(file_path):
    return np.eye(4)

def mesh2mask(vertices, triangles, shape, volsz, remove_temp_folder=True):

    assert min(volsz) > 0, 'The vol size needs to be greater than 0, if it is not, multiply by 10 until it is. Otherwise Mesh2MaskS.exe will fail.'

    folder_id = utils.create_temp_folder()
    # print(f'using folder {folder_id}')

    meshfile_path = f'{cur_folder}/temp/{folder_id}/mesh.mesh'
    maskfile_path = f'{cur_folder}/temp/{folder_id}/mesh.mask'

    utils.write_mesh(vertices, triangles, meshfile_path)

    cmd = f'{cur_folder}/bin/Mesh2MaskS.exe'
    check_call([cmd, meshfile_path, *map(str, shape), *map(str, volsz), '1', maskfile_path], stdout=DEVNULL, stderr=STDOUT)

    mask = utils.im2volume(maskfile_path, shape, np.uint8)
    mask[mask < 127] = 0
    mask[mask > 0] = 1

    if remove_temp_folder:
        utils.delete_temp_folder(folder_id)

    return mask

def points2marker(points, shape, marker_size=2, difference_color=False):
    contour = np.zeros(shape, dtype=np.int16)
    for i, p in enumerate(points):
        p = p.round().astype(np.int32)
        for x in range(p[0] - marker_size // 2, p[0] + marker_size // 2):
            for y in range(p[1] - marker_size // 2, p[1] + marker_size // 2):
                for z in range(p[2] - marker_size // 2, p[2] + marker_size // 2):
                    if x < 0 or x >= shape[0] or y < 0 or y >= shape[1] or z < 0 or z >= shape[2]: continue
                    contour[x, y, z] = i + 1 if difference_color else 1
    return contour

def mesh2contour(vertices, shape):
    contour = np.zeros(shape).astype(np.int16)
    for p in vertices:
        p = p.round().astype(np.int32)
        if p[0] < 0 or p[0] >= shape[0] or p[1] < 0 or p[1] >= shape[1] or p[2] < 0 or p[2] >= shape[2]: continue
        contour[int(round(p[0])), int(round(p[1])), int(round(p[2]))] = 1
    return contour

def mesh2contour_all(meshlist, shape):
    contour = np.zeros(shape).astype(np.int16)
    for i, vertices in enumerate(meshlist):
        for p in vertices:
            p = p.round().astype(np.int32)
            if p[0] < 0 or p[0] >= shape[0] or p[1] < 0 or p[1] >= shape[1] or p[2] < 0 or p[2] >= shape[2]: continue
            contour[int(round(p[0])), int(round(p[1])), int(round(p[2]))] = i + 1
    return contour

def parse_cdb(cdb_file):
    parser = CDBParser(cdb_file, keys=['surfs'])
    parser.parse()
    return parser.data

def point_register(source_points, target_points):

    xcenter = np.average(source_points, axis=0)
    ycenter = np.average(target_points, axis=0)
    x0 = source_points-xcenter
    y0 = target_points-ycenter

    H = np.matmul(x0.T, y0)
    U, W, V = np.linalg.svd(H)
    V = V.T
    # np.linalg.svd()

    mind = np.argmin(W)
    D = np.identity(3)
    D[mind][mind] = np.linalg.det(np.matmul(V, U.T))

    R = np.matmul(np.matmul(V, D), U.T)
    t = ycenter-np.matmul(R, xcenter.T).T

    y1 = np.matmul(R, source_points.T).T
    y1 += t
    mse = np.average(np.sqrt((target_points-y1)*(target_points-y1)))
    res = {
        'rotation': R,
        'translation': t,
        'mse':mse,
        'transformed': y1
    }

    return res