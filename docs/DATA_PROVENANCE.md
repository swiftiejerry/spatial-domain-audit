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

- **Section**: `Dataset11_Human_Lymph_Node_A1` from the SpatialGlue data release,
  originally derived from GEO accession `GSE263617`.
- **Supplied Annotation Boundary**: The SpatialGlue documentation notes manual
  annotation of tissue structures from H&E histology by a clinician collaborator.
  The local H5AD metadata does not document which external annotation revision
  supplied `obs['final_annot']`. We therefore treat `obs['final_annot']`
  explicitly as a *supplied reference annotation*, and all label-conditional
  metrics inherit this provenance boundary.

## Independent Retrieval Check (2026-11-19)

The official public archive was queried and only the required archive members
were retrieved into a private external-data area. The downloaded A1 RNA file
from the SpatialGlue archive is 56,563,170 bytes, and the paired ADT file is
1,139,080 bytes. Both were extracted from the official archive using its ZIP
member directory and passed the recorded per-member CRC32 check. The A1 RNA
matrix has shape `(3484, 18085)` and no `obs` columns. The ADT matrix has shape
`(3484, 31)`, no `obs` columns, and the same spot order as the RNA file.

The official A1 RNA matrix and the local labeled RNA matrix have identical
stored count values and spatial coordinates. Their spot names align after
removing the local `s1-` prefix. Three duplicated gene names in the official
file (`TBCE`, `HSPA14`, and `TMSB15B`) carry `-1` suffixes in the local file.
This verifies the matrix-level correspondence, but it does not establish the
origin of any local annotation column.

In the local A1 object, `final_annot` and `lab_lynn` differ at 59 of 3,484
spots. The observed transitions are `cortex` to `follicle` (31), `medulla cords`
to `medulla vessels` (20), `medulla sinuses` to `medulla vessels` (3),
`capsule` to `follicle` (2), `medulla cords` to `follicle` (2), and
`subcapsular sinus` to `follicle` (1). The retrieved official A1 RNA and ADT
files contain neither field, and the reviewed local materials contain no
revision record explaining these changes. The labels therefore remain
conditional supplied references. Their first-party lineage is unresolved.

## SpatialGlue Multimodal Run (2026-11-20)

The three-seed SpatialGlue analysis used the authors' public A1 result object's
30-dimensional `obsm['RNA_feat']` and `obsm['Pro_feat']` matrices as model
inputs. The corresponding raw RNA and ADT objects were independently checked
for exact spot order and spatial-coordinate alignment. No observation-label
field was copied into the model inputs, and labels were joined by spot ID only
after predictions had been saved. Each seed retained all 15 Leiden resolutions
from 0.1 to 1.5. A standalone CPU recount recomputed ARI, NMI, both binary
rare-F1 definitions, per-domain recall/F1/enrichment, and cluster counts for
all 45 saved predictions. Every saved score matched the independent recount.

At resolution 0.9, primary rare-F1 across seeds 0, 1, and 42 was
`0.1251 +/- 0.0896`. The three runs each captured at least two non-follicle
rare references in size-eligible predicted clusters. This is partial recovery
conditional on `final_annot`; it does not resolve the 59-spot difference from
`lab_lynn` or establish the four audit checks' operating characteristics.
Lightweight predictions, configs, and evaluations are stored under
`experiments/runs/spatialglue_A1_RNA_ADT/`. Embeddings, model weights, and raw
H5AD files remain in the private archive.

The public D1 RNA file retrieved from the SpaMosaic record is 30,721,462 bytes
and has shape `(3359, 18085)`. It contains `src` and `lab` observation columns.
An additional public mouse spleen RNA section is available at shape
`(2568, 32285)`, but the retrieved file has no observation labels, so it cannot
serve as a labeled cross-tissue evaluation without a separately documented
reference.

The A1 RNA and paired protein matrices were retrieved from the public SpatialGlue data release, and the D1 RNA matrix was retrieved from the public SpaMosaic data release. The source records list CC BY 4.0. The A1 matrices do not contain evaluation labels, and these data-source records do not establish the origin or revision history of `final_annot` in the separate labeled local object.

For A1, `final_annot` and `lab_lynn` differ at 59 spots. No reviewed first-party annotation file, revision record, or transformation script explains those changes. Evaluation scores that use `final_annot` therefore remain conditional on the supplied local reference labels. The official unlabeled RNA and ADT matrices confirm expression and modality inputs, not label provenance.

The retrieval inventory and ZIP-member checks are recorded in the private
external-data manifest. Raw H5AD files remain outside this repository.

## External Cross-Tissue Candidate Assessment (2026-11-19)

A public spatial DLPFC object was retrieved and opened for read-only inspection
of section-level manual layer labels. It contains 30 sections, with labels on
11,991 of 113,927 spots. Only one section is fully labeled, and its smallest
observed layer is 9.61% of that section. No individual section combines complete
labels with a class at or below 5%. Pooling three labeled sections yields a
4.60% white-matter share, but requires combining sections and masking 21 missing
labels. The data-specific reuse terms were not stated in the reviewed metadata
and remain unverified. No clustering or training was run, and this object is not
included in the repository.

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
