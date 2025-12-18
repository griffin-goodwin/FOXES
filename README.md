# FOXES: A Framework For Operational X-ray Emission Synthesis

This repository contains the code and resources for **FOXES**, a project developed as part of the _**Frontier Development Lab**_'s Heliolab Multimodal Flare Prediction Model Project.

### Abstract
Understanding solar flares is critical for predicting space weather, as their activity shapes how the Sun influences Earth and its environment. The development of reliable forecasting methodologies of these events depends on robust flare catalogs, but current methods are limited to flare classification using integrated soft X-ray emission that are available only from Earth’s perspective. This reduces accuracy in pinpointing the location and strength of farside flares and their connection to geoeffective events. 

In this work, we introduce a **Vision Transformer (ViT)**-based approach that translates Extreme Ultraviolet (EUV) observations into soft x-ray flux while also setting the groundwork for estimating flare locations in the future. The model achieves accurate flux predictions across flare classes using quantitative metrics. This paves the way for EUV-based flare detection to be extended beyond Earth’s line of sight, which allows for a more comprehensive and complete solar flare catalog. 

**Team**: Griffin Goodwin, Alison March, Jayant Biradar, Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Lorien Pratt

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
├── notebook_tests      # Visualization and testing notebooks
├── Dockerfile          # Docker configuration for environment reproducibility
└── requirements.txt    # Python dependencies
```

## Setup

### 1) Clone
```bash
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
```

### 2) Create an environment (conda or mamba)
```bash
mamba create -n foxes python=3.11 -y    # or: conda create -n foxes python=3.11 -y
mamba activate foxes                   # or: conda activate foxes
```

### 3) Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4) Docker Setup (Optional)
For a containerized environment, refer to [DOCKER_SETUP.md](DOCKER_SETUP.md) and [QUICKSTART_DOCKER.md](QUICKSTART_DOCKER.md).

## Citation
If you use this code or data in your work, please cite:

```aiexclude
@software{FOXES,
    title           = {{FOXES: A Framework For Operational X-ray Emission Synthesis}},
    institution     = {Frontier Development Lab (FDL), NASA Goddard Space Flight Center},
    repository-code = {https://github.com/griffin-goodwin/FOXES},
    version         = {v1.0},
    year            = {2025}
}
```

## Acknowledgement
This work is a research product of Heliolab (heliolab.ai), an initiative of the Frontier Development Lab (FDL.ai). FDL is a public–private partnership between NASA, Trillium Technologies (trillium.tech), and commercial AI partners including Google Cloud and NVIDIA.
Heliolab was designed, delivered, and managed by Trillium Technologies Inc., a research and development company focused on intelligent systems and collaborative communities for Heliophysics, planetary stewardship and space exploration.
We gratefully acknowledge Google Cloud for extensive computational resources, and NVIDIA Corporation for access to DGX Cloud and/or the Ada Lovelace L40 platform, enabled through NVIDIA and VMware.
