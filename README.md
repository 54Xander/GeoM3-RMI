# GeoM3-RMI

**Geometry-aware multimodal information fusion for RNA-small molecule interaction prediction**

GeoM3-RMI is a PyTorch framework for predicting RNA-small molecule interactions by integrating RNA sequence, secondary-structure topology, predicted tertiary geometry, small-molecule graphs, and semantic text representations. The model combines modality-specific encoders, adaptive multimodal fusion, cross-entity attention, and contrastive learning within a unified architecture.

![GeoM3-RMI framework](figures/framework.png)

## Repository structure

```text
GeoM3-RMI/
├── README.md
├── environment.yml
├── requirements.txt
├── training.py
├── model.py
├── create_data.py
├── utils.py
├── figures/
│   └── framework.png
├── data/
│   ├── 3D/
│   │   └── alphafold3/
│   │       ├── RNA.xlsx
│   │       ├── Molecule.xlsx
│   │       ├── RNA-Molecule.xlsx
│   │       └── PDB/
│   │           ├── <RNA_ID_1>.pdb
│   │           ├── <RNA_ID_2>.pdb
│   │           └── ...
│   └── processed/
├── pretrained_models/
│   └── biobert-base-cased-v1.2/
├── model/
└── result/
```

The main source files are:

| File | Description |
|---|---|
| `training.py` | Main training, validation, testing, checkpointing, and result-export script |
| `model.py` | GeoM3-RMI architecture and contrastive learning objectives |
| `create_data.py` | Data splitting and offline multimodal preprocessing |
| `utils.py` | Molecular graph construction, RNA structure processing, evaluation metrics, caching, and reproducibility utilities |

## Environment

The released Linux environment was exported to `environment.yml`. The principal software versions include:

| Software | Version |
|---|---:|
| Python | 3.11.11 |
| PyTorch | 2.10.0+cu128 |
| PyTorch Geometric | 2.6.1 |
| Transformers | 4.46.3 |
| RDKit | 2024.09.6 |
| NumPy | 1.26.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.5.1 |
| Biopython | 1.86 |
| CUDA runtime packages | 12.8 series |

A Linux system with an NVIDIA GPU and a driver compatible with CUDA 12.8 is recommended.

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate GeoM3-RMI
```

Verify the core installation:

```bash
python - <<'PY'
import torch
import torch_geometric
import transformers
import rdkit

print("PyTorch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)
print("Transformers:", transformers.__version__)
print("RDKit:", rdkit.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

`requirements.txt` is retained as a package-level snapshot of the exported environment. Because it contains Linux- and Conda-build-specific `file://` entries, `environment.yml` is the recommended file for reproducing the environment on another machine.

## Data preparation

Place the dataset under:

```text
data/3D/alphafold3/
```

The directory must contain three Excel workbooks and one PDB directory:

```text
data/3D/alphafold3/
├── RNA.xlsx
├── Molecule.xlsx
├── RNA-Molecule.xlsx
└── PDB/
    └── <RNA_ID>.pdb
```

### `RNA.xlsx`

Required columns:

| Column | Description |
|---|---|
| `RNA_ID` | Unique RNA identifier |
| `1D Sequence` | RNA nucleotide sequence |
| `Dot bracket` | RNA secondary structure in dot-bracket notation |
| `RNA information` | RNA text used by the semantic encoder |

### `Molecule.xlsx`

Required columns:

| Column | Description |
|---|---|
| `Small molecule_ID` | Unique small-molecule identifier |
| `SMILES` | Molecular SMILES representation |
| `Small molecule information` | Molecular text used by the semantic encoder |

### `RNA-Molecule.xlsx`

Required columns:

| Column | Description |
|---|---|
| `RNA_ID` | RNA identifier present in `RNA.xlsx` |
| `Small molecule_ID` | Small-molecule identifier present in `Molecule.xlsx` |
| `label` | Binary interaction label (`0` or `1`) |

### RNA tertiary structures

Each RNA PDB file must be stored as:

```text
data/3D/alphafold3/PDB/<RNA_ID>.pdb
```

The filename must match the corresponding value in the `RNA_ID` column.

### Data-quality requirements

Before training:

- confirm that every identifier in `RNA-Molecule.xlsx` occurs in the corresponding RNA or molecule workbook;
- confirm that all SMILES strings can be parsed by RDKit;
- confirm that RNA sequences and dot-bracket structures are correctly aligned;
- confirm that PDB filenames match the RNA identifiers;
- remove interaction outcomes, affinity measurements, and other label-revealing information from the semantic text fields.

## Pretrained semantic model

GeoM3-RMI uses BioBERT to encode RNA and small-molecule textual information. Place the pretrained model and tokenizer files under:

```text
pretrained_models/biobert-base-cased-v1.2/
```

The default command-line configuration uses this path for both semantic branches:

```text
--rna_semantic_model pretrained_models/biobert-base-cased-v1.2
--mol_semantic_model pretrained_models/biobert-base-cased-v1.2
```

The current implementation expects 768-dimensional semantic embeddings.

## Main experiment

The main experiment uses all RNA and small-molecule modalities:

- RNA sequence;
- RNA secondary-structure topology;
- atom- and residue-level RNA tertiary geometry;
- RNA semantic text;
- small-molecule graph structure;
- small-molecule semantic text.

It performs five-fold training with the default warm-start pair split. The validation set is constructed from the training portion of each fold, and the checkpoint with the highest validation AUC is retained.

### Run with the default configuration

```bash
python training.py
```

The command above is equivalent to:

```bash
python training.py \
  --dataset 3D/alphafold3 \
  --split_mode random \
  --seed 42 \
  --n_splits 5 \
  --val_size 0.1 \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.0005 \
  --weight_decay 0.00001 \
  --embed_dim 128 \
  --dropout 0.1 \
  --nhead 32 \
  --transformer_encoder_layer 1 \
  --k_value 3 \
  --max_seq_len 267 \
  --contrastive_temp 0.1 \
  --aux_weight_modal 0.25 \
  --aux_weight_interaction 0.1 \
  --scheduler_patience 1 \
  --scheduler_factor 0.7 \
  --early_stop_patience 20 \
  --pdb_dir PDB \
  --rna_semantic_model pretrained_models/biobert-base-cased-v1.2 \
  --mol_semantic_model pretrained_models/biobert-base-cased-v1.2
```

### Automatic preprocessing

When fold-specific processed files are absent, `training.py` automatically:

1. reads the three Excel workbooks;
2. generates the five data folds;
3. extracts BioBERT semantic embeddings;
4. constructs small-molecule graphs from SMILES;
5. constructs RNA secondary- and tertiary-structure graphs;
6. serializes the processed multimodal datasets;
7. starts fold-wise model training.

Processed files and graph caches are written under:

```text
data/processed/
```

To regenerate the data after changing the source workbooks, PDB structures, semantic model, or preprocessing code, remove the corresponding files under `data/processed/` before rerunning the experiment.

## Outputs

### Model checkpoints

The checkpoint with the highest validation AUC for each fold is saved as:

```text
model/3D_alphafold3_random_fold<fold>.pt
```

### Evaluation results

Epoch-level results are saved as:

```text
result/result_3D_alphafold3_random_fold<fold>_<timestamp>.csv
```

Each result file contains training, validation, and test values for:

- AUC;
- AUPR;

The reported fold-level result corresponds to the test metrics obtained at the epoch with the highest validation AUC.

## Reproducibility

The implementation sets deterministic seeds for Python, NumPy, PyTorch, CUDA, and data-loader workers. It also enables deterministic PyTorch operations and disables nondeterministic accelerated attention kernels when supported.

For manuscript-level reproducibility, retain the following information with every experiment:

```text
Source-code commit:
Dataset version:
RNA structure source:
Random seed:
Number of folds:
Python version:
PyTorch version:
PyTorch Geometric version:
CUDA version:
GPU model:
Training command:
```

The global SMILES cache is stored at:

```text
data/processed/cache/smiles_graph.pt
```

Delete this file before running a dataset containing a different set of molecules.
