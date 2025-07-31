from src.magnitude import (
    compute_comparison_magnitudes,
    compute_target_magnitudes,
    phase_lightcurve,
    count_table_trial,
    scan_count_periods,
    scan_sigma_periods,
    calculate_sigma
)
from src.data_loader import load_dataset
from src.error_functions import calculate_uncertainty_from_comparisons, calculate_total_magnitude_uncertainty, estimate_sigma_distribution, fit_amplitude_models
from src.utils import get_dataset_params, get_phase_params, plot_composite_lightcurve
import yaml
import numpy as np

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Start env: conda activate asteroid-env

# Load Data Testing
all_datasets = {
    d["name"]: load_dataset(d)
    for d in config["data"]["datasets"]
}

kobe20 = get_dataset_params(
    config,"kobe20",period=4.29112,offset=None,drop_indices=[3,7],system='johnson',observatory='k20',binning=False, mag_offset=0.08
    )

kobe21 = get_dataset_params(
    config,"kobe21",period=4.29112,offset=None,drop_indices=None,system='johnson',observatory='k21',binning=False, mag_offset=1.266
    )

pmo = get_dataset_params(
    config,"pmo",period=4.29112,offset=None,drop_indices=[0,2,20],system='sdss',observatory='pmo',binning=False, mag_offset=0
    )

params_list = [pmo, kobe20, kobe21]

# # Compute
sabine = compute_target_magnitudes(
    pmo['target_df'],pmo['comp_df'],pmo['lit_df'],
    drop_indices=pmo.get('drop_indices'),system=pmo['system'], return_mean=False
)
# print(sabine)

mag_err = calculate_uncertainty_from_comparisons(
    pmo["lit_df"],
    pmo['comp_df'],
    system=pmo['system'],
    drop_indices=[0,2,20],
    constrain_range=None,
    show=False,
    return_outliers=False
)
print(mag_err)

K20_terr = calculate_total_magnitude_uncertainty(
    comparison_error= mag_err,
    sabine_df=sabine,
    transform_errors=[0.045858, 0.03],
    verbose=False
)

K21_terr = calculate_total_magnitude_uncertainty(
    comparison_error= mag_err,
    sabine_df=sabine,
    transform_errors=[0.045858, 0.03],
    verbose=False
)

pmo_terr = calculate_total_magnitude_uncertainty(
    comparison_error= mag_err,
    sabine_df=sabine,
    transform_errors=[0.16497],
    verbose=False
)

terr_list = [pmo_terr, K20_terr, K21_terr]

#print(pmo_terr)

# phase = phase_lightcurve(
#     kobe20['target_df'], kobe20["comp_df"], kobe20['lit_df'], kobe20['utc_list'],
#     **get_phase_params(kobe20)
#         )
# print(phase)

# phase = phase_lightcurve(
#     pmo['target_df'], pmo["comp_df"], pmo['lit_df'], pmo['utc_list'],
#     **get_phase_params(pmo)
#         )
# print(phase)

counts = np.load("count_cache.npz")
count_tbl = counts['count_tbl']
avg_counts = counts["avg_counts"]
total_counts = counts["total_counts"]

# print(avg_counts)

trial_periods = np.arange(4.293, 4.295, 0.00001)

# df_scan, best_period, best_sigma = scan_sigma_periods(
#     trial_periods,
#     params_list=params_list,
#     avg_counts=avg_counts,
#     total_counts=total_counts,
#     verbose=True,
#     return_best=True
# )

# print(f"Best-fit period: {best_period:.6f} with sigma: {best_sigma:.4f}")
# print(df_scan)

# testing = calculate_sigma(4.2940, params_list, avg_counts, total_counts, return_table=True, shuffled_mags=True)
# print(testing)

# mu, sigma, fwhm, sigmas, best_tbl = estimate_sigma_distribution(
#     4.2940,
#     calculate_sigma=calculate_sigma,
#     params_list=params_list,
#     avg_counts=avg_counts,
#     total_counts=total_counts,
#     trials=500,
#     return_best_table=True,
#     plot=True
# )

# plot = plot_composite_lightcurve(
#     params_list=params_list,
#     phase_fct=phase_lightcurve,
#     period=4.29283,
#     period_err=0.00018,
#     terr_list=terr_list,
#     labels=['PMO 20200816','NHAO 20200824','NHAO 20211107'],
#     colors=['#1d61ad', '#c25925', '#18a835'],
#     markers=['o', 's', '^'],
#     amp_str="0.30",
#     jd0_str="JDo(UTC): 2459077.5",
#     fig_label="Figure",
#     fig_num=1,
#     title="665 Sabine Phased Light Curves",
#     err_frac=[0.04, 0.04, 0.03]
# )

# # print(plot)

# tbl = calculate_sigma(4.29286, params_list, avg_counts, total_counts, return_table=True, shuffled_mags=False)
# x, y = tbl['mean_phase'], tbl['mean_mag']
# sorted_indices = np.argsort(x)
# x_sorted = x[sorted_indices]
# y_sorted = y[sorted_indices]
# mean_err = (pmo_terr + K20_terr + K21_terr) / 3

# results = fit_amplitude_models(x_sorted, y_sorted, period=4.29286, mean_err=mean_err, deg=6, smooth=0.003)

# print(results)