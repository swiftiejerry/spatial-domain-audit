# Data Acquisition and Provenance

## Offline Reference Bundle (Distributed Here)

To allow CPU-only verification of all saved predictions, construct-validity
checks, attainability enumerations, and boundary-F1 / edge-retention audits
without redistributing the full raw expression matrix, this repository bundles:

```text
data/reference/
  A1_reference.npz            # 3,484 spot IDs, Visium coordinates, and supplied reference labels
  reference_annotation.tsv    # Tab-delimited reference label table keyed by spot_id
  spatial_edges_k10.npy       # Deduplicated undirected-union k=10 Euclidean k-NN edge index (2, 38624)
```

## Full Expression Input (External Placement)

Full model training and differential-expression marker screening require the
publicly deposited 10x Genomics Visium human lymph node RNA section placed at:

```text
data/raw/human_lymph_node_A1/adata_RNA.h5ad
```

## Required AnnData Schema

| Field | Specification |
|---|---|
| `X` | Raw RNA counts matrix, shape `(3484, 18085)` (spots by genes) |
| `obsm['spatial']` | Two-dimensional Visium array coordinates, shape `(3484, 2)` |
| `obs['final_annot']` | Ten-class reference domain annotation used as evaluation reference |

## Source and Annotation Provenance

- **Section**: `Dataset11_Human_Lymph_Node_A1` from the SpatialGlue data release
  (originally derived from GEO accession `GSE263617`; SpatialGlue processed
  archive DOIs: `10.5281/zenodo.7879713` and `10.5281/zenodo.10362607`, CC BY 4.0).
- **Supplied Annotation Boundary**: The SpatialGlue documentation notes manual
  annotation of tissue structures from H&E histology by a clinician collaborator.
  Because the local H5AD metadata does not preserve a cryptographic download
  manifest binding `obs['final_annot']` to a specific external release revision,
  `obs['final_annot']` is treated explicitly as a *supplied reference annotation*,
  and all label-conditional metrics inherit this provenance boundary.

## Verification Snippet

```python
import scanpy as sc

adata = sc.read_h5ad("data/raw/human_lymph_node_A1/adata_RNA.h5ad")
assert adata.shape == (3484, 18085)
assert adata.obsm["spatial"].shape == (3484, 2)
assert "final_annot" in adata.obs
print(adata.obs["final_annot"].value_counts())
```

Custom sections may be supplied via `--h5ad` and `--label-key`. Upstream
pseudo-label resolution follows the frozen `segment_function` rule (`0.2` for
`<= 500` spots, `0.5` for `<= 5,000` spots, and `0.8` otherwise) unless
overridden via `--pseudo-resolution`.
