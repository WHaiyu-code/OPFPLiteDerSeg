# OPFPLiteDerSeg
Rethinking Information Flow in Dermoscopic Lesion Segmentation: Optical Input Representation and Frequency-Selective Fusion
## Environment

The experiments were conducted under the following environment:

* **OS:** Ubuntu 18.04.6 LTS (Linux 5.4.0)
* **GPU:** NVIDIA Tesla V100-PCIE-32GB
* **Python:** 3.10.18
* **PyTorch:** 2.6.0+cu124
* **Torchvision:** 0.21.0
* **CUDA:** 12.4
* **cuDNN:** 9.1
* **NumPy:** 1.26.4
* **Albumentations:** 1.3.1
* **OpenCV:** 4.10.0
* **SciPy:** 1.15.3
* **scikit-image:** 0.19.2
* **segmentation-models-pytorch:** 0.5.0
* **timm:** 1.0.22

### Installation

We recommend creating a dedicated Conda environment:

```bash
conda create -n metadseg python=3.10 -y
conda activate metadseg
```

Install PyTorch with CUDA support:

```bash
pip install torch==2.6.0 torchvision==0.21.0
```

Install the main dependencies:

```bash
pip install \
    numpy==1.26.4 \
    albumentations==1.3.1 \
    opencv-python==4.10.0 \
    scipy==1.15.3 \
    scikit-image==0.19.2 \
    segmentation-models-pytorch==0.5.0 \
    timm==1.0.22
```

To verify the PyTorch and CUDA configuration:

```bash
python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA:', torch.version.cuda); \
print('cuDNN:', torch.backends.cudnn.version()); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The reference environment reports:

```text
PyTorch: 2.6.0+cu124
CUDA: 12.4
cuDNN: 90100
CUDA available: True
GPU: Tesla V100-PCIE-32GB
```
