# PG‑TAF: Perceptually‑Guided Graphs & Type‑Aware Fusion for Enhanced GRIP++ Trajectory Prediction

<!-- PG‑TAF architecture image -->
![PG‑TAF architecture](./Modified%20grip%20architecture.drawio.png)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PG‑TAF is a set of principled architectural refinements to the **GRIP++** framework for **multi‑agent trajectory prediction**.  
The model boosts accuracy by introducing **perceptually‑guided elliptical interaction graphs**, **agent‑type embeddings with mid‑fusion**, and **optimized recurrent & GCN components**—all implemented in **PyTorch**.

> **Primary benchmark:** *ApolloScape Trajectory* dataset.

---

## 🚀 Key Enhancements

| Area | PG‑TAF Improvement |
|------|-------------------|
| **Graph Construction** | *Elliptical* interaction neighbourhoods (e.g. **15‑5‑0‑12**) instead of purely radial zones. |
| **Feature Representation** | <ul><li>Learnable **agent‑type embeddings** (5 types → 8 dims).</li><li>**Mid‑fusion**: 2‑stage Conv2D to combine type & kinematics *after* initial motion feature learning.</li></ul> |
| **Dynamic Edge Weights** | Learnable importance matrix passed through <br>*LayerNorm → LeakyReLU → Softmax* for stable weighting. |
| **GCN Layers** | Linear transform → graph norm → *(optional)* residual → LayerNorm → LeakyReLU. |
| **Recurrent Module** | **Bidirectional GRUs** with weighted sum of fwd/back states. |
| **Training Best‑Practices** | BatchNorm after Conv2D and before activation (à la ResNet). |

---

## 📂 Repository Layout

```text
PG‑TAF/
├── data/                       # Place ApolloScape splits here
├── models/                     # Saved checkpoints
├── layers/                        # Model, dataset & utils
│   ├── conv1.py
│   ├── graph_conv_block.py
│   ├── graph_operation_layer.py
│   ├── graph.py
│   └── seq2seq.py
├── scripts/                    # Helper bash / python scripts
├── main.py                     # Train / test entry‑point
├── model.py                     # Train / test entry‑point
├── data_process.py             # Circular‑neigh preprocessing
├── data_process_ellipse.py     # Elliptical‑neigh preprocessing
├── utils.py     # Elliptical‑neigh preprocessing
├── xin_feeder_baidu.py     # Elliptical‑neigh preprocessing
└── README.md
```

---

## 🛠️ Installation

### Prerequisites

* Python ≥ 3.8  
* PyTorch ≥ 1.7.1 (install the build matching your CUDA)  
* `numpy`, `scipy`, `argparse`, `pickle` (std‑lib), etc.

### Quick Start

```bash
# 1. Clone
git clone https://github.com/mahmoudahroug/PG-TAF.git
cd PG-TAF

# 2. (Recommended) create env
conda create -n pg_taf_env python=3.8 -y
conda activate pg_taf_env

# 3. Install deps  (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy
```

> Provide a `requirements.txt` or `environment.yml` for exact versions if replicability is critical.

---

## 📈 Dataset Setup – ApolloScape Trajectory

1. **Download** the ApolloScape trajectory split from the official source.  
2. **Extract** into `./data/`, e.g.

```text
PG‑TAF/
└── data/
    ├── prediction_train/
    └── prediction_test/
```

3. **Pre‑process**

```bash
# Circular (original GRIP++) neighbourhood
python data_process.py

# Elliptical neighbourhood (PG‑TAF)
python data_process_ellipse.py
```

These scripts cache the graph tensors & normalised trajectories expected by the model.

---

## 🏋️‍♀️ Training

```bash
python main.py   --config configs/pg_taf.yaml   --train   --gpus 0
```

Key flags:

* `--config` – YAML with model & optimisation hyper‑params.  
* `--train` / `--test` – toggles mode.  
* `--gpus` – comma‑sep GPU ids.

---

## 📊 Evaluation

1. **Generate predictions**

```bash
python main.py   --config configs/pg_taf.yaml   --test   --checkpoint models/best_pg_taf.pth
```

This writes `.txt` files in the format required by the official ApolloScape evaluation.

2. **Official metric**

Run the *ApolloScape Trajectory Prediction Evaluation* script (see their repo).  
PG‑TAF reports **WSADE** (Weighted Sum of ADE)—lower is better.

---

## 🧪 Key Ablation Results (ApolloScape · WSADE ↓)

| ID | Incremental Config | WSADE |
|----|--------------------|------:|
| 2  | Unofficial GRIP++ baseline | 1.2449 |
| 4  | + LeakyReLU & Softmax on importance matrix | 1.2301 |
| 10 | + Mid‑fusion type emb. & elliptical (15‑5‑0‑15) | **1.1878** |
| 13 | + Bidirectional GRU (hidden=30, weighted sum) | 1.1950 |

---

## 📜 Citation

```bibtex
@mastersthesis{Dahroug2025PGTAF,
  author = {Mahmoud Dahroug},
  title  = {Perceptually-Guided Graphs and Type-Aware Fusion (PG-TAF): Enhancing GRIP++ for Multi-Agent Trajectory Prediction},
  school = {German University in Cairo},
  year   = {2025},
  note   = {Bachelor Thesis}
}
```

---

## 🤝 Acknowledgements

PG‑TAF builds upon the excellent **GRIP++** framework and insights from many trajectory‑prediction works.  
Thanks to the ApolloScape authors for providing a comprehensive benchmark.

---

© 2025 Mahmoud Dahroug — released under the MIT License
