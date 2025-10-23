# NASA HelioLab 2025 Multimodal Modal Flare Prediction Project

This repository serves as the main repository for _**Frontier Development Lab**_
's Heliolab Multimodal Flare Prediction Model Project.

**Team**: Alison March, Jayant Biradar, Griffin Goodwin

**Faculty**: Christopher Schirninger, Robert Jarolim, Angelos Vourlidas, Lorien Pratt

<p align="center">
  <a href="https://postimg.cc/dZ2XqvNZ">
    <img src="https://i.postimg.cc/pr4tJLDC/Flare-logo.png" alt="Flare-logo.png" width="150" />
  </a>
<a href="https://postimg.cc/ygXtKsbg">
    <img src="https://i.postimg.cc/d1zY80RB/FOXES-logo.png" alt="FOXES-logo.png" width="150" />
</p>



This challenge aims to develop a virtual 4π GOES soft X-ray (SXR) flare monitor using front- and far-side EUV images to enable continuous 
solar flare prediction across the entire solar disc, enhancing space weather forecasting. By merging this virtual instrument output with ensemble 
forecasting and integrating observational data, the system will continuously update the probability distribution of imminent solar eruptions across 
the Sun.

Solar flares, sudden bursts of energy from the Sun, pose a significant threat to our technological infrastructure, including satellites, power grids, 
and GPS. Real-time monitoring of these events is crucial for mitigating their impact. The GOES satellite series provides essential data on a flare's soft X-ray flux, 
a key indicator of its strength. However, GOES data is an integrated measurement, meaning it tells us how strong a flare is but not its specific location on the Sun's surface. 
To pinpoint a flare's origin, we need complementary data, such as the extreme ultraviolet (EUV) images from NASA's Solar Dynamics Observatory (SDO). 
The current reliance on two separate systems (GOES for strength and SDO for location) is a major limitation, preventing a unified approach to flare forecasting and real-time response.

To solve this problem, we developed “FOXES”, a novel machine learning model that can extract both the strength and location of a solar flare from SDO EUV images alone. The solution is based on a Vision Transformer (ViT) architecture, a class of models that treats images similarly to how large language models process sentences. By segmenting solar images into smaller “patches”, like words in a sentence, the model learns complex patterns between these patches. This allows it to identify which regions of the sun are most active and accurately translate the visual information from SDO images into a predicted soft X-ray flux value, essentially creating a “virtual GOES” instrument. Training the model on a combination of SDO and GOES data, the model consistently shows high accuracy, particularly for the most powerful and dangerous flares.

FOXES represents a significant step forward in space weather forecasting. By consolidating two separate data streams into a single, unified system, it provides a powerful tool for real-time flare detection. The model can not only predict flare strength with high fidelity, even when traditional GOES data is missing, but also provide interpretable heatmaps that pinpoint a flare's precise location. This capability is a game-changer for mission planners and space weather teams. Furthermore, because FOXES operates on image data, it can be applied to solar images from viewpoints beyond Earth's line of sight, such as those from the STEREO mission. This will enable us to create the first-ever comprehensive multimodal flare catalog and estimate the strength of events on the far side of the Sun. This revolutionary leap forward will be critical for protecting astronauts on future missions to deep space, such as to Mars, and will enhance our ability to prepare for solar events before they impact Earth.

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" style="margin: 4px;">
  </a>
  <a href="https://sunpy.org">
    <img src="https://img.shields.io/badge/powered%20by-SunPy-F07820.svg?style=flat" alt="Powered by SunPy" style="margin: 4px;">
  </a>
  <a href="https://www.astropy.org">
    <img src="https://img.shields.io/badge/powered%20by-Astropy-2C3E50.svg?style=flat" alt="Powered by Astropy" style="margin: 4px;">
  </a>
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Google Cloud" style="margin: 5px;">
  </a>
  <img src="https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green" alt="PyCharm" style="margin: 5px;">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" style="margin: 5px;">
  <img src="https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515" alt="YAML" style="margin: 5px;">
  <img src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white" alt="Conda" style="margin: 5px;">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" style="margin: 5px;">
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white" alt="Weights & Biases" style="margin: 5px;">
  <img src="https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white" alt="Git" style="margin: 5px;">
</p>


<p align="center">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" style="margin: 5px;">
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" style="margin: 5px;">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="Matplotlib" style="margin: 5px;">
  <img src="https://img.shields.io/badge/Seaborn-4C9ABA.svg?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn" style="margin: 5px;">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" style="margin: 5px;">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" style="margin: 5px;">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" style="margin: 5px;">
</p>


## Repository Structure

``` 
Project FOXES
├── data                # Data cleaning and preprocessing procedures
├── download            # Various datasets download methods(SDO, SXR, STEREO, Solar Orbiter, Flare event locations)
├── notebook_tests      # Visualization/testing ipynb
├── utils               # Extra stuff(utility) 
├── forecasting         # Main code directory for model forecasting
│   ├── data_loaders    # SDO_AIA dataloader and sxr normalization procedure
│   ├── inference       # This script automates the process of running inference and evaluation for multiple model checkpoints.
│   ├── models          # Contains various models such as patch-based Vision Transformer with 2D positional encoding and local attention to predict solar soft X-ray flux.
└── └── training        # Contains train.py and callback.py scripts
        ├──train.py     # This script trains a Vision Transformer (ViTLocal) to predict solar soft X-ray flux from AIA imagery. 
        │                 It loads a YAML config, prepares AIA–GOES data, computes optional class weights, and logs training with Weights & Biases. 
        │                 The script automatically detects CPU/GPU hardware, runs training, and saves the best and final models with full experiment tracking.
        └──callback.py  # Logs model predictions and visualize attention maps during validation, comparing true vs. predicted soft X-ray flux and displaying detailed Transformer attention patterns in Weights & Biases.
```

## Datasets

| **Name**                                                                                                                                                                                                                            | **Description** | **Granularity & Source**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Solar Dynamics Observatory (SDO)** <br> *[Pesnell, W. D., Thompson, B. J., & Chamberlin, P. C. (2012)](https://doi.org/10.1007/s11207-011-9841-3).The Solar Dynamics Observatory (SDO)*                                           | A NASA mission launched in 2010 to observe the Sun’s atmosphere at high temporal and spatial resolution using instruments like AIA, HMI, and EVE.| SDO/AIA EUV data is downloaded at 1-minute cadence using the JSOC DRMS series in the six wavelengths 94 Å, 131 Å, 171 Å, 193 Å, 211 Å, and 304 Å.<br><br>**Access/download:** <br/>Available from [JSOC](http://jsoc.stanford.edu/ajax/exportdata.html) and [VSO](https://sdac.virtualsolar.org/cgi/search). For programmatic access, use the [SunPy Fido](https://docs.sunpy.org/en/stable/guide/acquiring_data/fido.html) interface.<br><br> *Machine learning–ready versions available on request |
| **Geostationary Operational Environmental Satellite (GOES/XRS)** <br> *[Garcia, H. A. (1994)](https://link.springer.com/article/10.1007/BF00681100). Temperature and emission measure from GOES soft X-ray measurements.*           | The GOES spacecraft series carries the X-ray Sensor (XRS), which continuously monitors solar soft X-ray flux in two wavelength bands (0.5–4 Å and 1–8 Å), providing near-real-time flare detection and classification. | 1-minute cadence soft X-ray flux data in two wavelength bands (0.5–4 Å and 1–8 Å).<br><br> **Access/download:** Data were accessed via [SunPy’s Fido](https://docs.sunpy.org/en/stable/guide/acquiring_data/fido.html) client, which queries the [VSO](https://sdac.virtualsolar.org/cgi/search) backend from [NOAA’s NCEI](https://www.ngdc.noaa.gov/stp/satellite/goes/) archive, downloading files per spacecraft and concatenating them into time series. |
| **Heliophysics Event Knowledgebase (HEK)** <br> *[Hurlburt, N. E., et al. (2012)](https://arxiv.org/pdf/1008.1291). Heliophysics Event Knowledgebase for the Solar Dynamics Observatory (SDO).*                                     | The HEK system provides an integrated database of solar events, including flares, coronal mass ejections, and active regions, identified from multiple instruments such as SDO/AIA and GOES. | Flare catalog entries were obtained through [SunPy’s HEK interface](https://docs.sunpy.org/en/stable/code_ref/net.html#module-sunpy.net.hek), by specifying date ranges, flare class thresholds, and observatory filters. The resulting data were converted into structured CSV files for downstream analysis and machine learning applications. |
| **Solar TErrestrial RElations Observatory (STEREO/SECCHI)** <br> *[Howard, R. A., et al. (2008)](https://link.springer.com/article/10.1007/s11214-008-9341-4). Sun–Earth Connection Coronal and Heliospheric Investigation (SECCHI).* | The twin STEREO spacecraft (Ahead and Behind) carry the SECCHI instrument suite, which observes the solar corona and heliosphere in multiple wavelengths, providing stereoscopic views of CMEs and large-scale coronal structures. | STEREO/SECCHI EUV and coronagraph data were retrieved via [SunPy’s Fido](https://docs.sunpy.org/en/stable/guide/acquiring_data/fido.html) client through the [VSO](https://sdac.virtualsolar.org/cgi/search) backend. Datasets from the EUVI (EUV Imager) were accessed per spacecraft and wavelength, then temporally aligned and processed for use in heliospheric modeling. |
| **Solar Orbiter (EUI – Extreme Ultraviolet Imager)** <br> *[Rochus, P., et al. (2020)](https://doi.org/10.1051/0004-6361/201936663e). The Solar Orbiter EUI instrument: The Extreme Ultraviolet Imager.*                            | The ESA–NASA Solar Orbiter mission carries the EUI instrument suite to image the solar atmosphere in the extreme ultraviolet at unprecedented spatial resolution and variable solar latitudes, enabling studies of small-scale heating and dynamic processes in the corona. | Solar Orbiter/EUI level-2 data products were accessed via the [ESA Solar Orbiter Archive (SOAR)](https://soar.esac.esa.int/soar/). Datasets were queried by observation date and wavelength, downloaded as FITS files, and preprocessed into standardized time series for integration with other solar observatories. |

## Setup

### 1) Clone
```
git clone https://github.com/griffin-goodwin/FOXES.git
cd FOXES
```
### 2) Create an environment (conda or mamba )

```
mamba create -n foxes python=3.11 -y    # or: conda create -n foxes python=3.11 -y
mamba activate foxes                   # or: conda activate foxes
```

### 3) Install Python dependencies
```
pip install -r requirements.txt
```
*(The repo includes a requirements.txt at the top level.)*


### 4) Configure paths & options
Open yaml files and set any local paths (data directories, cache/output locations, etc.).

### Quickstart Usage
For more information regarding how to run data pipeline processes, please see [README_pipeline.md](https://github.com/griffin-goodwin/FOXES/blob/main/data/README_pipeline.md) under FOXES/data.
For information regarding running batch evaluation process, see [README_batch_evaluation.md](https://github.com/griffin-goodwin/FOXES/blob/main/forecasting/inference/README_batch_evaluation.md) under FOXES/forecasting/inference.


### Best Practices

To ensure reproducibility, consistency, and efficient workflow management across experiments, follow these best practices:

### Environment & Dependencies
- Always create a dedicated virtual environment (e.g., `conda` or `mamba`) to isolate dependencies.
- Freeze installed packages with:
```
pip freeze > requirements-lock.txt
```

## Citation
If you use this code or data in your work, please cite:

```aiexclude
@software{FOXES,
    title           = {{FOXES: Forecasting Solar X-ray Emission using SDO/AIA and GOES/XRS}},
    institution     = {Frontier Development Lab (FDL), NASA Goddard Space Flight Center},
    repository-code = {https://github.com/griffin-goodwin/FOXES},
    version         = {v1.0},
    year            = {2025}
}
```

## Acknowledgement:
This work is a research product of Heliolab (heliolab.ai), an initiative of the Frontier Development Lab (FDL.ai). FDL is a public–private partnership between NASA, Trillium Technologies (trillium.tech), and commercial AI partners including Google Cloud and NVIDIA.
Heliolab was designed, delivered, and managed by Trillium Technologies Inc., a research and development company focused on intelligent systems and collaborative communities for Heliophysics, planetary stewardship and space exploration.
We gratefully acknowledge Google Cloud for extensive computational resources, and NVIDIA Corporation for access to DGX Cloud and/or the Ada Lovelace L40 platform, enabled through NVIDIA and VMware.
