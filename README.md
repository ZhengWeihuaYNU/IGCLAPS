# scCLAPS
Source codes of scCLAPS: a graph contrasitve learning method with adaptive positive sampling for clustering scRNA-seq data.
# Requirements:
python---3.12

torch---2.3.0

numpy---1.26.4

scikit-learn---1.5.2

torch_geometric---2.6.0

scanpy---1.10.3

munkres---1.1.4

dgl---2.1.0

h5py---3.11.0

# Usage:
Just run 
`python main.py
`
in a command line.

For your own data, please make sure that the data is stored in h5 format, containing X as a cell * gene raw count matrix and Y as the real cell type labels.

Please note that the installation of torch_geometric may require some dependency packages, before installing torch_geometric, please run 
`pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
`
