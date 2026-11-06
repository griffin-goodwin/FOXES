```{image} _static/foxes-logo.png
:alt: FOXES logo: a fox made of solar image patches against an orange Sun.
:class: foxes-hero-logo
```

# FOXES

```{container} foxes-subtitle
Framework For Operational X-ray Emission Synthesis
```

FOXES estimates the Sun's full-disk soft X-ray flux (analogous to GOES) based on a provided seven channel stack of SDO/AIA
images. It also provides an estimate of the flux coming from individual regions through the resulting flux maps.
The original method and its evaluation are described in
[Goodwin et al. (2026), *ApJ* 1003, 48](https://doi.org/10.3847/1538-4357/ae5e4a).

If you're new here, [Getting Started](python_api.md) walks you through
installing the package and making your first prediction. After that,
[Models and Checkpoints](models.md) covers the available weights and
[Results](results.md) shows how the current model performs on the held-out
test set.
If you want to retrain the model or run batch jobs from a clone of the repo,
see [Command-Line Workflows](workflows.md).

```{toctree}
:maxdepth: 1
:caption: Documentation

python_api
models
results
uncertainties
workflows
api_reference
citation
```

[Paper](https://doi.org/10.3847/1538-4357/ae5e4a) ·
[Source Repository](https://github.com/griffin-goodwin/FOXES) ·
[PyPI](https://pypi.org/project/helio-foxes/) ·
[Notebook Walkthrough](https://github.com/griffin-goodwin/FOXES/blob/main/tutorials/heliofoxes_demo/heliofoxes.ipynb) ·
[Live Demo](https://huggingface.co/spaces/griffingoodwin04/FOXES-model) ·
[Dataset](https://huggingface.co/datasets/griffingoodwin04/FOXES-Data)
