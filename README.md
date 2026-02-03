# FOXES: A Framework For Operational X-ray Emission Synthesis

This repository contains the code and resources for **FOXES**, a project developed as part of the _**Frontier Development Lab**_'s Heliolab Multimodal Flare Prediction Project.

### Abstract
The solar soft X-ray (SXR) irradiance is a long-standing proxy of solar activity, used for the classification of flare strength. As a result, the flare class, along with the SXR light curve, are routinely used as the primary input to many forecasting methods, from coronal mass ejection speed to energetic particle output. However, the SXR irradiance lacks spatial information leading to dubious classification during periods of high activity, and is  applicable only for observations from Earth orbit, hindering forecasting for other places in the heliosphere. This work introduces the Framework for Operational X-ray Emission Synthesis (FOXES), a Vision Transformer-based approach for translating Extreme Ultraviolet (EUV) spatially-resolved observations into SXR irradiance predictions. The model produces two outputs: (1) a global SXR flux prediction and (2) per-patch flux contributions, which offer a spatially resolved interpretation of where the model attributes SXR emission. This paves the way for EUV-based flare detection to be extended beyond Earth's line of sight, allowing for a more comprehensive and reliable flare catalog to support robust, scalable, and real-time forecasting, extending our monitoring into a true multiviewpoint system.

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Viacheslav Sadykov, Lorien Pratt

## Repository Structure

```text
FOXES
├── data                # Data cleaning and preprocessing procedures
├── download            # Datasets download methods (SDO, SXR, STEREO, Solar Orbiter, etc.)
├── forecasting         # Main code directory for model forecasting
│   ├── data_loaders    # AIA dataloader and sxr normalization procedure
│   ├── inference       # Inference and evaluation scripts
│   ├── models          # Vision Transformer model definitions
│   └── training        # Training scripts and callbacks
├── misc                # Personal utility scripts (gitignored)
├── notebook_tests      # Visualization and testing notebooks
├── Dockerfile          # Docker configuration for environment reproducibility
├── foxes.yml           # Conda environment file
└── requirements.txt    # Python dependencies
```

## Setup

### 1) Clone
```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
```

### 2) Create an environment

**Option A — pip:**
```bash
conda create -n foxes python=3.11 -y
conda activate foxes
pip install -r requirements.txt
```

**Option B — conda (full environment):**
```bash
conda env create -f foxes.yml
conda activate foxes
```

### 4) Docker Setup (Optional)
For a containerized environment, refer to [DOCKER_SETUP.md](DOCKER_SETUP.md) and [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md).

## Citation
If you use this code or data in your work, please cite:

```aiexclude
@software{FOXES,
    title           = {{FOXES: A Framework For Operational X-ray Emission Synthesis}},
    institution     = {Frontier Development Lab (FDL)},
    repository-code = {https://github.com/griffin-goodwin/FOXES},
    version         = {v1.0},
    year            = {2026}
}
```

## Acknowledgement
This work is a research product of Heliolab (heliolab.ai), an initiative of the Frontier Development Lab (FDL.ai). FDL is a public–private partnership between NASA, Trillium Technologies (trillium.tech), and commercial AI partners including Google Cloud and NVIDIA.
Heliolab was designed, delivered, and managed by Trillium Technologies Inc., a research and development company focused on intelligent systems and collaborative communities for Heliophysics, planetary stewardship and space exploration.
We gratefully acknowledge Google Cloud for extensive computational resources, and NVIDIA Corporation for access to DGX Cloud and/or the Ada Lovelace L40 platform, enabled through NVIDIA and VMware.
