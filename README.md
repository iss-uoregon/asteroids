# Asteroid Light Curve Analysis

This repository provides a modular pipeline for processing, phasing, and modeling asteroid light curves collected from multiple observatories. Designed around datasets from Pine Mountain Observatory and Nishi-Harima Astronomical Observatory's study of asteroid 665 Sabine, the tools can be adapted to similar photometric studies involving Johnson and Sloan filters and GAIA reference stars.

The pipeline supports composite light curve generation, amplitude modeling, and uncertainty estimation, and is broken into modules that can be run independently or through a consolidated main.py.

## Features
- Epoch folding and bin-based light curve compositing
- Fourier, spline, and polynomial amplitude fitting
- Photometric uncertainty propagation using field stars and literature mags
- Monte Carlo–based period error estimation via magnitude shuffling
- Modular design for easy adaptation to new asteroid datasets

## Requirements
See `requirements.txt`.

## Setup Environment
Use the following to create and activate a Python environment:
```bash
conda create -n asteroid-env python=3.10
conda activate asteroid-env
pip install -r requirements.txt
```

## Usage Overview
### Config
Create a user defined config.yaml following the blueprint of the included Sabine config. 
This defines paths to:
    - Literature magnitudes
    - Comparison star photometry
    - Target object photometry
    - UTC timestamps for each observation file
    - Specific dataset filters

For each dataset, ensure filenames are consistently prefixed (e.g., kobe20_lit.txt, kobe20_comp.txt).

### Imports
Import the primary functions into your main.py, or equivalent file, following the example modules. For ease of use, keep your working file outside of the src folder.

### Parameter Dictionaries
Use get_dataset_params from src/utils.py to define parameters for each dataset, and bundle them in a list inside src/params_list.py:
    ```python
        example = get_dataset_params(
        config,"kobe20",period=4.29112,offset=None,
        drop_indices=[3,7],system='johnson',observatory='k20',
        binning=False, mag_offset=0.08
        )
    params_list = [example, example2, ...]```

### Lightcurve Module
These functions generate phased light curves with uncertainties, and plot composite lightcurves at given periods:

    calculate_total_magnitude_uncertainty :
    Combines uncertainty contributions from comparison stars, target photometry, and filter conversions (e.g., Gaia → SDSS g′ or Johnson V).

    phase_lightcurve :
    Folds and bins each dataset at a trial period to prepare it for modeling or composite plotting.

    plot_composite_lightcurve :
    Creates publication-ready light curve plots for submission (e.g., to the Minor Planet Bulletin). Typically used once a best-fit period is determined.

See drivers/magnitude_module.py for usage.

### Folding Module
These functions scan and fold datasets over trial periods, scanning for phase alignment :

    scan_count_periods :
    Precomputes normalized bin weights used in folding routines. Should be cached before running folding analysis.

    calculate_sigma :
    Calculates alignment strength (sigma-from-mean) at a single period. Can shuffle magnitudes for uncertainty testing.

    scan_sigma_periods :
    Iteratively runs calculate_sigma across a range of periods. Returns a sigma-vs-period curve for best-fit estimation.

    fit_amplitude_models :
    Fits and overlays polynomial, spline, and Fourier amplitude models at a trial period. Useful for deriving peak-to-peak brightness ranges.

See drivers/folding_module.py for usage.

### Shuffle Module
Monte Carlo routines to estimate period uncertainty:

    calculate_sigma (with shuffled_mags=True) :
    Shuffles magnitudes within each dataset before folding.

    estimate_sigma_distribution :
    Repeats calculate_sigma over multiple trials at a fixed period. Fits a Gaussian to the resulting sigma histogram and returns the one-sigma width as an uncertainty.

Example usage can be found in drivers/shuffle_module.py.

Note: On sparse datasets, aliasing and cycle count ambiguity may dominate true error, limiting shuffle accuracy.

## Citation
If you use or adapt this pipeline, please cite the original authors or relevant publication describing the Sabine analysis:

## Future Improvements
- Add user-defined filter conversion support
- Generalize photometry input handling beyond Gaia references
- Speed up large-scale period scans using multiprocessing
