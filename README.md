# IIVFormer: Factorized Intra-View and Inter-View Attention for Multi-View 2D-to-3D Human Pose Lifting

Official PyTorch implementation of **IIVFormer: Factorized Intra-View and Inter-View Attention for Multi-View 2D-to-3D Human Pose Lifting**.

IIVFormer first models joint relationships inside each camera view with intra-view attention, then models information across camera views with inter-view attention. The model maps `(B, V, J, 2)` inputs to root-relative `(B, 1, J, 3)` poses, where `B` is the batch size, `V` is the number of views and `J` is the number of joints. For Human3.6M, `V=4` and `J=17`.

## Repository

```bash
git clone https://github.com/JerryPengNJ/IIVFormer.git
cd IIVFormer
```

Main files:

```text
IIVFormer/
├── common/
│   ├── IIVFormer.py       # Intra-view and inter-view Transformer model
│   ├── data_utils.py      # Dataset wrapper, split and normalization
│   ├── h36m_dataset.py    # Human3.6M skeleton and camera metadata
│   ├── loss.py            # MPJPE and P-MPJPE metrics
│   ├── camera.py          # Camera-coordinate utilities
│   └── Logger.py          # Training logger
├── main.py                # Training and validation entry point
├── evaluate.py            # Protocol 1 and Protocol 2 evaluation
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

Training and evaluation are configured through command-line arguments in `main.py` and `evaluate.py`. The default settings used for the reported Human3.6M results are listed below.

## Environment

The reference environment is:

| Component | Version |
|---|---:|
| Python | 3.13.5 |
| PyTorch | 2.8.0+cu129 |
| TorchVision | 0.23.0+cu129 |
| CUDA | 12.9 |
| cuDNN | 9.10.2 |

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

The code calls CUDA directly during training and model construction, so a CUDA-capable GPU is required for the provided commands.

## Dataset

The released training and evaluation scripts use Human3.6M data prepared in the VideoPose3D format. Follow [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to generate the 3D annotations and 2D detections.

The following files are required for the default CPN experiment:

```text
data/
├── data_3d_h36m.npz
└── data_2d_h36m_cpn_ft_h36m_dbb.npz
```

The data directory is ignored by Git. It can be placed outside the repository and linked into the project:

```bash
ln -s /path/to/Data data
```

This repository provides the complete Human3.6M training and evaluation pipeline with CPN 2D detections.

## Preprocessing

Preprocessing is performed online by `main.py`, `evaluate.py` and `common/data_utils.py`:

1. Load `positions_2d` from the 2D detection archive and the Human3.6M 3D poses from `data_3d_h36m.npz`.
2. Trim extra 2D frames so every camera sequence has the same length as its 3D motion-capture sequence.
3. Concatenate the four synchronized camera views for each frame.
4. Convert 3D poses from meters to millimeters and make them root-relative by subtracting joint 0.
5. Compute the 2D mean and standard deviation from the training subjects only, then use those statistics to normalize both training and test inputs.
6. Reshape the normalized input to `(B, V, J, 2)` and the target to `(B, 1, J, 3)`. For Human3.6M, `V=4` and `J=17`.

The default split is:

| Split | Subjects |
|---|---|
| Train | S1, S5, S6, S7, S8 |
| Test | S9, S11 |


## Default Configuration

### Model

| Option | Value |
|---|---:|
| Number of views | 4 |
| Number of joints | 17 |
| Input channels | 2 |
| Embedding dimension | 32 |
| Intra-view Transformer depth | 4 |
| Inter-view Transformer depth | 4 |
| Attention heads | 8 |
| Output coordinates | 3 |

The constructor argument `num_view` is set to `4` by the scripts and represents the number of camera views in this implementation.

### Training

| Option | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.0004 |
| Batch size | 1024 |
| Epochs | 100 |
| Loss | MPJPE |
| DataLoader workers | 4 |
| Random seed | 42 |

## Random Seed

Both entry points expose `--seed`, with a default value of `42`. The scripts seed Python, NumPy, PyTorch and all CUDA devices. They also set:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Exact bitwise results can still vary across GPU models, CUDA versions and PyTorch builds. Report the software and hardware environment together with the seed when publishing results.

## Pretrained Weights

The pretrained Human3.6M CPN checkpoint and training log are available in the [shared Google Drive folder](https://drive.google.com/drive/folders/1ldILNj8Lon6gdx3_sq4hEW0Y8f7u2a3Z?usp=sharing).

| Artifact | Description | Download | Size | SHA-256 |
|---|---|---|---:|---|
| `cpn.pth` | Human3.6M CPN checkpoint: 30.9 mm MPJPE (P1), 24.0 mm P-MPJPE (P2) | [cpn.pth](https://drive.google.com/drive/folders/1ldILNj8Lon6gdx3_sq4hEW0Y8f7u2a3Z?usp=sharing) | 38,284,671 bytes | `bca9b8c6dc0a998af05c03ab67d6e391bab369c6694c70f917c9ebabb2209d60` |
| `cpn.log` | Human3.6M CPN training log | [cpn.log](https://drive.google.com/drive/folders/1ldILNj8Lon6gdx3_sq4hEW0Y8f7u2a3Z?usp=sharing) | 13,990 bytes | `7d2b1094f1de6b80143cda6f61a6ce81ac792684320b431e403dfab215b3d095` |

The links above open the shared folder; select the artifact with the listed filename. Verify downloaded files with:

```bash
sha256sum cpn.pth cpn.log
```

Place `cpn.pth` in the repository root or pass its full path through `--model_path`. The released checkpoint results are 30.9 mm under Protocol 1 and 24.0 mm under Protocol 2.

## Training

Run the default Human3.6M CPN experiment:

```bash
python main.py \
  --data_3d data/data_3d_h36m.npz \
  --data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz \
  --batch_size 1024 \
  --epochs 100 \
  --lr 0.0004 \
  --seed 42
```

The script writes training logs to `cpn.log` and saves the best state dictionary as `cpn.pth`.

## Evaluation

Protocol 1 reports MPJPE in millimeters:

```bash
python evaluate.py \
  --data_3d data/data_3d_h36m.npz \
  --data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz \
  --subjects "[\"S9\", \"S11\"]" \
  --model_path cpn.pth \
  --protocol p1 \
  --seed 42
```

Protocol 2 reports P-MPJPE after rigid Procrustes alignment:

```bash
python evaluate.py \
  --data_3d data/data_3d_h36m.npz \
  --data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz \
  --subjects "[\"S9\", \"S11\"]" \
  --model_path cpn.pth \
  --protocol p2 \
  --seed 42
```

The evaluator prints the error for each action and the mean over all Human3.6M actions.

## Reproduction Commands

A complete default run is:

```bash
git clone https://github.com/JerryPengNJ/IIVFormer.git
cd IIVFormer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu129
ln -s /path/to/Data data

python main.py \
  --data_3d data/data_3d_h36m.npz \
  --data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz \
  --batch_size 1024 --epochs 100 --lr 0.0004 --seed 42

python evaluate.py \
  --data_3d data/data_3d_h36m.npz \
  --data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz \
  --subjects "[\"S9\", \"S11\"]" \
  --model_path cpn.pth --protocol p1 --seed 42
```

## Visualization

The visualization setup follows [MHFormer](https://github.com/Vegetebird/MHFormer).

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{peng2025iivformer,
  author  = {Guozheng Peng},
  title   = {IIVFormer: Factorized Intra-View and Inter-View Attention for Multi-View 2D-to-3D Human Pose Lifting},
  journal = {The Visual Computer},
  year    = {2025},
  note    = {Submitted}
}
```
