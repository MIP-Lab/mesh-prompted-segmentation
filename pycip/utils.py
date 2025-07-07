import os
import time
import shutil
import numpy as np
import scipy.io
import random
import pyvista
try:
    import nibabel as nib
except:
    pass

cur_folder = __file__.replace('utils.py', '').replace('\\', '/')[: -1]

def create_temp_folder():
    folder_id = f'temp_{time.time()}_{random.randint(0, 1000000)}'
    os.mkdir(f'{cur_folder}/temp/{folder_id}')
    return folder_id

def delete_temp_folder(folder_id):
    shutil.rmtree(f'{cur_folder}/temp/{folder_id}')

def mat2str(matrix):
    out = ''
    for row in matrix:
        out += ' '.join([f'%.5f' % x for x in row]) + '\n'
    return out

def vec2str(vector, dtype='float'):
    if dtype == 'float':
        return ' '.join([f'%.5f' % x for x in vector])
    
    return ' '.join([f'%.0f' % x for x in vector])

# the transpose order is important, equivalent to matlat reshape(data, shape)
def im2volume(im_file, shape, dtype):
    data = np.fromfile(im_file, dtype=dtype)
    img = np.transpose(data.reshape([shape[2], shape[1], shape[0]]), [2, 1, 0])
    return img

def volume2im(volume):
    im = np.transpose(volume, [2, 1, 0]).flatten()
    return im

def write_mesh(vertices, triangles, file):
    data = [0, vertices.shape[0], triangles.shape[0], 255, 0, 0]
    with open(file, 'wb') as f:
        f.write(np.array(data, dtype='int32').tobytes())
        f.write(vertices.flatten().astype(np.float32).tobytes())
        f.write(triangles.flatten().astype(np.int32).tobytes())
    return 0

def get_orientation_from_affine(affine):
    pivot = np.abs(affine[: 3, : 3]).argmax(axis=0).tolist()
    orient = [pivot.index(0) + 1, pivot.index(1) + 1, pivot.index(2) + 1]
    for i, n in enumerate(orient):
        if (n == 1 or n == 2) and affine[i, n - 1] > 0:
            orient[i] *= -1
        if (n == 3) and affine[i, n - 1] < 0:
            orient[i] *= -1
    return orient

def change_orientation(image, to_orient, from_orient=None):
    if from_orient is None:
        from_orient = get_orientation_from_affine(image.affine)
    affine = image.affine.copy()
    data = np.array(image.dataobj).copy()
    abs_to_orient = np.abs(to_orient).tolist()
    abs_from_orient = np.abs(from_orient).tolist()
    axes_order = [0, 0, 0]
    for i in range(3):
        axes_order[abs_to_orient[i] - 1] = abs_from_orient[i] - 1
    data = np.transpose(data, axes_order)
    affine[: 3, : 3] = affine[: 3, axes_order]
    temp_orient = [
        from_orient[abs_from_orient.index(abs_to_orient[0])],
        from_orient[abs_from_orient.index(abs_to_orient[1])],
        from_orient[abs_from_orient.index(abs_to_orient[2])]
        ]
    
    for i in range(3):
        o1, o2 = temp_orient[i], to_orient[i]
        assert abs(o1) == abs(o2)
        if o1 != o2:
            data = np.flip(data, axis=i)
            affine[:, i] *= -1
    
    return nib.Nifti1Image(data, affine)

def read_mesh(file):
    with open(file, 'rb') as f:
        mid = np.fromfile(f, 'int32', 1)[0]
        numverts = np.fromfile(f, 'int32', 1)[0]
        numtris = np.fromfile(f, 'int32', 1)[0]
        n = np.fromfile(f, 'int32', 1)[0]
        if n == -1:
            orient = np.fromfile(f, 'int32', 3)
            dim = np.fromfile(f, 'int32', 3)
            sz = np.fromfile(f, 'float32', 3)
            color = np.fromfile(f, 'int32', 3)
        else:
            color = np.fromfile(f, 'int32', 2)
        vertices = np.fromfile(f, 'float32', numverts * 3)
        vertices = vertices.reshape((-1, 3))
        triangles = np.fromfile(f, 'int32', numtris * 3)
        triangles = triangles.reshape((-1, 3))
        return vertices, triangles

def read_mesh_asPyVista(file):
    vertices, triangles = read_mesh(file)
    faces_4d = np.zeros((triangles.shape[0], 4))
    faces_4d[:, 1:4] = triangles
    faces_4d[:, 0] = 3
    faces_flatten = faces_4d.flatten().astype(np.int32)
    mesh = pyvista.PolyData(vertices, faces_flatten)
    return mesh

def to_pyvista(vertices, triangles):
    faces_4d = np.zeros((triangles.shape[0], 4))
    faces_4d[:, 1:4] = triangles
    faces_4d[:, 0] = 3
    faces_flatten = faces_4d.flatten().astype(np.int32)
    mesh = pyvista.PolyData(vertices, faces_flatten)
    return mesh

def write_mesh_asPyVista(mesh, file):
    vertices = mesh.points
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]
    write_mesh(vertices, triangles, file)

def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], scipy.io.matlab.mio5_params.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)