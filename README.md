# TransGen

## Project Overview

De novo enzyme design remains challenging because abstract catalytic logic is difficult to encode directly into amino acid sequences. Existing approaches often rely on homology search, random mutation, or coarse functional conditioning, resulting in low efficiency and substantial manual intervention. To address these limitations, we introduce TransGen, a generative model that uses dilated convolutions and directly integrates reaction-level catalytic changes into protein sequence generation. The model is trained and evaluated on UDP-glycosyltransferases (UGTs), enzymes crucial in plant metabolism. Each reaction is encoded as a single Reaction Vector, defined as the difference between aggregated product and reactant embeddings. Incorporating the Reaction Vector through Adaptive Layer Normalization (AdaLN) conditions residue-level sequence patterns, while alignment of latent representations with the ESM-2 knowledge space promotes the biological plausibility of generated sequences. Docking analysis suggests that the generated sequences preserve pocket-level structural compatibility with the target reactions. By generating protein sequences in response to reaction-specific chemical signals, TransGen provides a promising framework for controllable enzyme sequence design.
## Key Features

- **Protein Feature Extraction**: Extract high-dimensional embeddings from protein sequences using pre-trained models like ESM-2 and ProtT5
- **Molecular Processing**: SMILES parsing and molecular graph construction based on RDKit
- **Data Preprocessing**: Tools for FASTA file processing, dataset splitting, and offset calculation
- **Graph Neural Network Support**: Protein contact map construction and edge weight computation

## Project Structure

```
TransGen/
├── utils/
│   ├── protein_init.py      # Protein initialization and feature extraction (ESM-2, ProtT5)
│   ├── ligand_init.py       # Ligand/molecule initialization and feature extraction
│   ├── split_data.py        # Dataset splitting and FASTA processing tools
│   ├── split.py             # SMILES tokenization and Transformer components
│   ├── dataset.py           # Dataset loaders
│   ├── build_vocab.py       # Vocabulary building
│   ├── metrics.py           # Evaluation metrics
│   └── trainer.py           # Trainer implementation
├── evodiff/                 # EvoDiff diffusion model module
├── config/                  # Configuration files
├── analysis/                # Analysis scripts
└── examples/                # Example code
```

## Installation Requirements

### Dependencies

```bash
pip install torch>=2.0.1
pip install transformers
pip install esm
pip install rdkit
pip install biopython
pip install torch-geometric
pip install h5py
pip install pandas
pip install numpy
pip install tqdm
```

### Pre-trained Models

The following pre-trained models are required:
- **ESM-2**: `esm2_t33_650M_UR50D` (automatically downloaded from HuggingFace)
- **ProtT5**: [prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)

## Quick Start

### 1. Protein Feature Extraction

```python
from utils.protein_init import ProteinInit

# Initialize protein feature extractor
protein_extractor = ProteinInit()

# Extract protein features
sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
features = protein_extractor.protein_init(sequences)

# Returned features include:
# - seq: protein sequence
# - token_representation: residue-level embeddings from ProtT5
# - token_representation_esm: embeddings from ESM-2
# - edge_index: contact graph edge indices
# - edge_weight: edge weights
```

### 2. SMILES Tokenization

```python
from utils.split import split

# Tokenize SMILES string
smiles = "CCO"
tokenized = split(smiles)
print(tokenized)  # Output: "C C O"
```

### 3. Dataset Splitting

```python
from utils.split_data import split, get_offsets

# Split FASTA dataset
split("data/proteins.fasta", "data/splits.json")

# Calculate offsets (for efficient data loading)
get_offsets("data/proteins.fasta", "data/offsets.npz")
```

## Core Modules

### utils/protein_init.py

**Main Class:**
- `ProteinInit`: Main protein feature extraction class integrating ESM-2 and ProtT5 models

**Key Functions:**
- `get_T5_model()`: Load ProtT5 model
- `esm_extract()`: Extract protein embeddings and contact maps using ESM-2
- `contact_map()`: Construct protein contact graphs
- `seq_feature()`: Calculate amino acid physicochemical properties

**Feature Dimensions:**
- ProtT5 embeddings: 1024 dimensions/residue
- ESM-2 embeddings: 1280 dimensions/residue
- Amino acid physicochemical features: 19 dimensions/residue (one-hot + 7 physicochemical properties)

### utils/split.py

**Main Features:**
- Intelligent SMILES tokenization (handles two-character elements like Cl, Br, Si)
- Transformer components (FFN, LayerNorm, etc.)
- SMILES validity verification

### utils/split_data.py

**Main Features:**
- FASTA file flattening
- Random dataset splitting (train/test/valid)
- Offset calculation (for streaming large FASTA files)
- Sequence length statistical analysis

## Data Processing Pipeline

1. **Prepare FASTA Files**: Ensure correct sequence format without special characters
2. **Data Cleaning**: Remove duplicate sequences, handle abnormal characters (replace U/B/Z/O with X)
3. **Dataset Splitting**: Use `split()` function to generate train/test/valid splits
4. **Calculate Offsets**: Use `get_offsets()` to generate index files for faster data loading
5. **Feature Extraction**: Use `ProteinInit` to extract protein embeddings and contact maps

## Important Notes

- **GPU Requirements**: ESM-2 and ProtT5 models require significant GPU memory; CUDA devices recommended
- **Sequence Length**: Sequences longer than 1000 amino acids will be truncated
- **Memory Optimization**: Use half precision (float16) to reduce memory usage
- **Batch Processing**: Batch feature extraction is recommended for better efficiency
