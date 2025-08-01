import numpy as np
import time
from src.params_list import pmo, kobe20, kobe21, params_list
from src.magnitude import scan_count_periods, calculate_sigma, scan_sigma_periods
from src.error_functions import fit_amplitude_models
from drivers.magnitude_module import pmo_terr, k20_terr, k21_terr

"""
Driver file for scanning periods, computing phase-folding sigma metrics,
and modeling amplitude curves using asteroid Sabine datasets.

Steps:
1. Generate (or load) bin count normalization cache
2. Calculate sigma at a fixed period
3. Scan for best-fitting period
4. Model the amplitude of phased light curve
"""
start = time.time()

def main():
    # 1. Loading a save path for the normalization scheme used in calculate_sigma.
    # Recommended: if no save path created, run this once to cache binning counts and speed up future sigma evaluations.
    USE_CACHE = True
    if USE_CACHE:
        counts = np.load("count_cache.npz")
        count_tbl = counts['count_tbl']
        avg_counts = counts["avg_counts"]
        total_counts = counts["total_counts"]
    else:
        count_tbl, avg_counts, total_counts = scan_count_periods(
            params_list, 4.291, 4.297, 0.00001, save_path="count_cache.npz"
        )
    
    # 2. Example calculate_sigma function call using Sabine data.
    folded_sigma_example = calculate_sigma(4.2940, params_list, avg_counts, total_counts, return_table=True, shuffled_mags=False)
    # print(folded_sigma_example)
    
    # 3. Example folding routine using Sabine data. Scanning range should match range used in count_tbl generation.
    trial_periods = np.arange(4.291, 4.297, 0.00001)
    
    df_scan, best_period, best_sigma = scan_sigma_periods(
        trial_periods,
        params_list=params_list,
        avg_counts=avg_counts,
        total_counts=total_counts,
        verbose=True,
        return_best=True
    )
    # print(f"Best period: {best_period}h at sigma: {best_sigma}")
    
    # 4. Example of amplitude modeling after a best_period has been identfied for the Sabine data. Returns fit results and a figure.
    tbl = calculate_sigma(best_period, params_list, avg_counts, total_counts, return_table=True, shuffled_mags=False)
    x, y = tbl['mean_phase'], tbl['mean_mag']
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    mean_err = (pmo_terr + k20_terr + k21_terr) / 3
    results = fit_amplitude_models(x_sorted, y_sorted, period=best_period, mean_err=mean_err, deg=6, smooth=0.003)
    # print(results)
    
    print(f"Elapsed time: {time.time() - start:.2f} sec")

if __name__ == "__main__":
    main()
