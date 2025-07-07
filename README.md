# mesh-prompted-segmentation

This repository contains code accompanying the paper titled "Shape-Guided Anatomy Segmentation using Mesh Prompt."
![method_overview](https://github.com/user-attachments/assets/0f39272a-9a33-43b1-a3aa-9d61ad7268ad)

## 1. Requirements

- **OS:** Windows or Linux  
- **Python:** ≥ 3.8
- **Nibabel:** ≥ 4.0.1
- **PyTorch:** ≥ 1.12  
- **PyTorch3D:** ≥ 0.7.0 *(used for Chamfer loss)*  
- **PyVista:** ≥ 0.43.0 and **Matplotlib:** ≥ 3.7.5 *(for visualization)*  
- **jupyter_core:** ≥ 5.7.2 *(for reproducing statistical analysis)*

---

The repository includes key components of the proposed method, such as the data loader, vector field attention network, training and inference scripts, and a Jupyter notebook for reproducing the results presented in the paper.

Please note that some path and dependency configurations are not yet fully resolved, and some data preprocessing scripts are missing, so the code may not run out of the box. We recommend using it as a reference for understanding the proposed methodology.

A fully functional version is currently under development. For questions regarding the paper or this repository, please contact dingjie.su@vanderbilt.edu.
