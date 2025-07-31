from src.magnitude import (
    compute_comparison_magnitudes,
    compute_target_magnitudes,
    phase_lightcurve,
    scan_count_periods
)
from src.data_loader import load_dataset, load_all_datasets
from src.error_functions import calculate_uncertainty_from_comparisons, calculate_total_magnitude_uncertainty
from src.utils import convert_utc_to_fractional_day, compute_light_time_offset, get_dataset_params
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# kobe20 = get_dataset_params(
#     config,"kobe20",period=4.29112,offset=None,drop_indices=[3,7],system='johnson',observatory='k20',binning=True, mag_offset=0.08
#     )

# kobe21 = get_dataset_params(
#     config,"kobe21",period=4.29112,offset=None,drop_indices=None,system='johnson',observatory='k21',binning=True, mag_offset=1.266
#     )

# pmo = get_dataset_params(
#     config,"pmo",period=4.29112,offset=None,drop_indices=[0,2,20],system='sdss',observatory='pmo',binning=True, mag_offset=0
#     )

# params_list = [pmo, kobe20, kobe21]

# count_tbl, avg_counts, total_counts = scan_count_periods(params_list, 4.291, 4.297, 0.00001, save_path="count_cache.npz")

import sys
print("\n".join(sys.path))