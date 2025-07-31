import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.magnitude import compute_comparison_magnitudes, compute_target_magnitudes
from src.utils import convert_utc_to_fractional_day, convert_to_sdss_g

def phase_lightcurve(
    target_df, comp_df, lit_df, utc_list,
    period, offset, bin_width=0.05, drop_indices=None, system="generic",
    observatory="generic", mags=False, show=False, binning=False
):
    T_lookup = {"k21": 365 + 15 + 30 + 31 + 7, "k20": 8, "pmo": 0}
    T = T_lookup.get(observatory, 0)

    # Compute fractional days
    fractional_days = convert_utc_to_fractional_day(utc_list, offset, T)
    
    # Compute target magnitudes
    raw_mags = compute_target_magnitudes(target_df, comp_df, lit_df, drop_indices, system, return_mean=True)
    
    # print(raw_mags)
    # print(fractional_days)

    # Create table
    df = pd.DataFrame({
        "Asteroid Single Mag": raw_mags,
        "Date": fractional_days
    })

    # print(df["Date"].head(10))
    # print(len(df['Date']))

    # Phase calculation
    rotation_days = period / 24
    df["Phase"] = (df["Date"] % rotation_days) / rotation_days
    df["bin"] = (df["Phase"] / bin_width).astype(int)

    # Optional SDSS g' conversion
    if observatory in ["k21", "k20"]:
        df["g Mag"] = convert_to_sdss_g(df["Asteroid Single Mag"])

    # Binning
    n_bins = int(1.0 / bin_width)
    group_cols = {
        "sum_mag": ("Asteroid Single Mag", "sum"),
        "count": ("Asteroid Single Mag", "size"),
        "phase": ("Phase", "mean")
    }
    if "g Mag" in df.columns:
        group_cols["g_mag"] = ("g Mag", "sum")
    
    binned = df.groupby("bin").agg(**group_cols).reindex(range(n_bins), fill_value=0)
    binned["Asteroid Mean Mag"] = binned["sum_mag"] / binned["count"]
    binned["Asteroid Mean Mag"].fillna(0, inplace=True)

    # Plotting
    if show:
        if mags:
            plt.scatter(df["Phase"], df["Asteroid Single Mag"], s=5)
            plt.ylabel("Sabine Mean Magnitude")
        else:
            plt.scatter(df["Phase"], df["Asteroid Single Mag"] - df["Asteroid Single Mag"].mean(), s=5)
            plt.ylabel("Magnitude Differences")
        plt.xlabel("Phase")
        plt.title(f"Phased Lightcurve: {observatory.upper()}")
        plt.xlim(-0.05,1.05)
        plt.gca().invert_yaxis()
        plt.show()
    
    if binning:
        return binned
    return df