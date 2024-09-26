import scanpy as sc
def preprocess(data, top_genes= 3000, n_comps= 100):
    adata = sc.AnnData(data)
    adata.var_names_make_unique()
    #quality control:
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts= 5)
    sc.pp.normalize_total(adata, target_sum= 1e6)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=top_genes)
    #dimensionality reduction:
    sc.pp.pca(adata, n_comps= n_comps)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added="clusters", resolution= 1.0, flavor= 'igraph')
    return adata
