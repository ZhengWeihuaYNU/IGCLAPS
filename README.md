# IGCLAPS
Source codes of IGCLAPS: an interpretable graph contrastive learning method with adaptive positive sampling for scRNA-seq data analysis.
# Requirements:
cuda---12.1

python---3.12

torch---2.3.0

numpy---1.26.4

pandas--2.2.3

scikit-learn---1.5.2

torch_geometric---2.6.0

scanpy---1.10.3

munkres---1.1.4

dgl---2.1.0

h5py---3.11.0

igraph---0.11.8
# Note:
Other packages can be installed by 'pip install xxx', while dgl should be installed by running 
`pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html`. Besides, after installing torch_geometric, please run `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html` to install dependencies.  
# Usage:
Just run 
`python main.py
`
in a command line or run test_script.ipynb in jupyter lab.

For your own data, please make sure that the data is stored in .h5 format containing raw count matrix 'X' with cells as rows and genes as columns. 

# Data availability:
PBMC 4k: [source](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k?)

Darmanis: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE67835)

LaManno: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76381)

Baron human: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)

Baron mouse: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)

Muraro: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241)

Bladder: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097)

Adam: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94333)

Zanini: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147668)

Colquitt: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE150486)

Young: [source](https://ega-archive.org/datasets/)

Chen: [source](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544)
