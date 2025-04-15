# IGCLAPS
Source codes of IGCLAPS: an interpretable graph contrastive learning method with adaptive positive sampling for scRNA-seq data analysis.
# Requirements:
cuda---12.1

python---3.12

torch---2.3.0

numpy---1.26.4

scikit-learn---1.5.2

torch_geometric---2.6.0

scanpy---1.10.3

munkres---1.1.4

dgl---2.1.0

h5py---3.11.0

# Note:
Other packages can be installed by "pip install xxx", while dgl should be installed by running "pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html". Besides, after installing torch_geometric, please run "pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html" to install dependent packages.  
# Usage:
Just run 
`python main.py
`
in a command line.

For your own data, please make sure that the data is stored in h5 format, containing X as a cell * gene raw count matrix and Y as the real cell type labels.

Please note that the installation of torch_geometric may require some dependency packages, before installing torch_geometric, please run 

`pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
`
