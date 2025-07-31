# Asteroid Light Curve Analysis

This repository contains tools for phasing, modeling, and analyzing asteroid light curves from multiple datasets. In its current iteration, these tools rely on data taken using Johnson and Sloan filtersets, alongside GAIA comparison star references. This program is divided into several modules that tackle different analysis and processing procedures, and an example format that was used for Pine Mountain Observatory's 665 Sabine project is included for reference. Each module can be ran individually, or consolidated into a single main.py file - however, several components iterate hundreds to thousands of times, so they have been divided

## Features
- Epoch folding with binned and weighted analysis
- Fourier, spline, and polynomial amplitude fitting
- Field star photometry error and Monte Carlo shuffle period error estimations
- Modular pipeline for multi-source lightcurve datasets

## Requirements
See `requirements.txt`.

## Setup Environment
```bash
conda create -n asteroid-env python=3.10
conda activate asteroid-env
pip install -r requirements.txt
```

## Usage
### Config
Create a user defined config.yaml following the blueprint of the Sabine data. For each subset of data (literature mags, comparison star photometry, target photometry, and fileset UTC-hour time stamps) ensure that the leading term in the dataset title is consistent - ie: kobe20 in the included example.

### Imports
Import the primary functions into your main.py, or equivalent file, following the example main.py file. For ease of import, keep your working file outside of the src folder.

### Parameters
Create parameter dictionaries for each relevant dataset and bundle them in a list for use in folding algorithms:
    ```example = get_dataset_params(
        config,"dataset_header",period=4.29112,offset=None,drop_indices=[3,7],system='johnson',observatory='dataset_desig',binning=False, mag_offset=0.08
        )
    params_list = [example, example2, ...]```

### Lightcurve Module
These functions take in target and comparison star photometric data, alonside relevant literature magnitudes, to produce phased, composited lightcurves with target magnitude uncertainties. Notably, these functions rely on use of GAIA DR2 comparison stars, whose conversion factors have been hard-coded into Johnson V and SDSS g' passbands using literature equations. Future iterations of these functions will prompt user input for any necessary conversions. An example template is included in Module 1.

Relevant function calls:
    calculate_uncertainty_from_comparisons : creates a comparative spread of comparison stars and their ability to accurately measure the magnitude of the remaining field stars. Function fits a Gaussian to the spread of deviations from literature values, and extracts the 1-sigma width of the distribution, producing an estimated error for the collective comparison stars.
    
    calculate_total_magnitude_uncertainty : calculates the total magnitude uncertainty for a given file set taking the three main sources of error into account: uncertainties from filter set transformations, observed photometric uncertainties of the target, and comparative uncertainties of each reference field star and their photometric quality using results from calculate_uncertainty_from_comparisons.

    phase_lightcurve : produces a table of binned or individualized magnitude and phase data for a given file set at an input trial period. This function is essential in procedures for other modules, but can also be individually called to assess the phasing results of a dataset.

    plot_composite_lightcurve : plotting function that takes in each relevant file set and produces a composite, phased lightcurve at a trial period. Function is setup to output in the standard expected of the Minor Planet Bulletin. This function is best used after the folding routine explored in Module 2 as identified the best-fit period for target, but can be ran at any time for visualization purposes.

### Folding Module
These functions are responsible for the epoch folding routine, associated error functions, and 
