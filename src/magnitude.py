import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import transform_magnitude_system, convert_utc_to_fractional_day, convert_to_sdss_g, get_phase_params

def compute_comparison_magnitudes(lit_df, comp_df, drop_indices=None, system="gaia"):
    """
    Calculate each comparison star's magnitude based on the rest of the field.

    Parameters:
    - lit_df : Dataframe of literature Gaia stars and their magnitudes
    - comp_df : Dataframe of measured comparison stars
    - drop_indices : List of any star indices to drop from each frame

    Returns:
    - Dataframe of comparison star magnitudes as calculated by the rest of the field.
    """

    df_lit = lit_df.copy()
    df_meas = comp_df.copy()

    # Drop row of lit_df and column of comp_df containing photometric data of a given field star
    if drop_indices:
        valid = [i for i in drop_indices if i < len(df_lit)]
        df_lit = df_lit.drop(index=valid).reset_index(drop=True)
        df_meas = df_meas.drop(columns=df_meas.columns[valid]).reset_index(drop=True)

    # Transform lit_df Gaia magnitudes into specified photometric system
    Mag = transform_magnitude_system(df_lit.iloc[:,0],df_lit.iloc[:,1],df_lit.iloc[:,2], system=system)
    comparisons = {}

    # Calculate magnitude of a given comparison star via every other field star
    for i, Ma in enumerate(Mag):
        Ia = df_meas.iloc[:, i]
        for j in range(len(Mag)):
            Mb = Ma + 2.5 * np.log10(Ia / df_meas.iloc[:, j])
            comparisons[f"Star {i} v {j}"] = Mb

    return pd.concat(comparisons, axis=1)

def compute_target_magnitudes(
    target_df,
    comp_df,
    lit_df,
    drop_indices=None,
    system='johnson',
    target='Sabine',
    return_star_mags=False,
    return_mean=False
):
    """
    Computes target magnitudes using comparison star magnitudes.
    Assumes df_target contains only a single flux column and a 'mag_err' column.

    Parameters:
    - target_df : Dataframe containing photometric data of the target asteroid.
    - comp_df : Dataframe containing photometric data of field comparison stars
    - lit_df : Dataframe containing literature magnitude data of comparison stars.
    - drop_indices : List of star indices to drop from comp_df and lit_df.
    - system : String referencing the photometric system to transform lit_df magnitudes into.
    - target : String name of the target asteroid.
    - return_star_mags : If True, returns a dataframe of comparison star magnitudes after photometric conversions.
    - return_mean : If True, returns the average of each comparison star's magnitude calculation of the target asteroid per file.

    Returns:
    - Dataframe containing the magnitude of the target asteroid as calculated from the field comparison stars.
    """

    # Clean df_lit and df_comps
    df_meas = comp_df.drop(columns=['source'], errors='ignore')
    
    if drop_indices:
        valid_positions_lit = [i for i in drop_indices if i < len(lit_df)]
        valid_positions_meas = [i for i in drop_indices if i < df_meas.shape[1]]

        if valid_positions_lit:
            idx_to_drop = lit_df.index[valid_positions_lit]
            lit_df = lit_df.drop(index=idx_to_drop).reset_index(drop=True)

        if valid_positions_meas:
            df_meas = df_meas.drop(columns=df_meas.columns[valid_positions_meas])

    # Get flux column from df_target (assumed to be first column)
    df_target_flux = target_df.iloc[:, 0]

    # Compute average magnitudes of comparison stars
    comp_mags = compute_comparison_magnitudes(lit_df, df_meas, system=system)

    n_stars = df_meas.shape[1]
    star_mags = []

    for i in range(n_stars):
        cols = [col for col in comp_mags.columns if col.endswith(f"v {i}")]
        per_frame_mags = comp_mags[cols].mean(axis=1)
        star_mags.append(per_frame_mags.mean())

    star_mags = np.array(star_mags)

    # Compute Sabine magnitudes from each comparison star
    asteroid_df = pd.DataFrame()
    for i, Ma in enumerate(star_mags):
        flux = df_meas.iloc[:, i]
        Mb = Ma + 2.5 * np.log10(flux / df_target_flux)
        asteroid_df[f'{target} Mag v star {i}'] = Mb

    # Compute magnitude uncertainty if 'mag_err' is available
    if 'mag_err' in target_df.columns:
        flux_err = target_df['mag_err']
        asteroid_df['Mag Uncertainty'] = 1.0857 * flux_err / df_target_flux

    # Clean NaNs if any (e.g., due to dropped target rows)
    asteroid_df = asteroid_df.dropna(subset=asteroid_df.columns).reset_index(drop=True)

    # Return requested output
    if return_star_mags:
        return comp_mags
    elif return_mean:
        return asteroid_df.filter(regex='Sabine').mean(axis=1)
    else:
        return asteroid_df
    

def phase_lightcurve(
    target_df, comp_df, lit_df, utc_list,
    period, offset, bin_width=0.05, drop_indices=None, system="generic",
    observatory="generic", mags=False, show=False, binning=False, mag_offset=0,
    shuffled_mags=False
):
    """
    Function that produces phased dataframes of asteroid data. Can be binned into specified phase bins
    or individually calculated. Can also shuffle asteroid magnitudes for Monte Carlo Shuffle Test uncertainty measurements.

    Parameters:
    - target_df : Dataframe of photometric data of target asteroid.
    - comp_df : Dataframe of photometric data of field comparison stars.
    - lit_df : Dataframe of literature magnitudes of field comparison stars.
    - utc_list : List of UTC times. Expects list to only contain HH:MM:SS data.
    - period : Float period estimation to phase against.
    - offset : Float magnitude offset determined from light travel time to target asteroid.
    - bin_width : Float < 1 that determines bin sizing. Default of 0.05 for 20 phase bins.
    - drop_indices : List of star indices to drop from comp_df and lit_df for magnitude calculations.
    - system : String photometric system to transform comparison stars into.
    - observatory : observation time specified key for establishing phase zero point.
    - mags : If True, the output figure plots raw magnitude data.
    - show : If True, plots a phased lightcurve at the input period.
    - binning : If True, returns a binned dataframe.
    - mag_offset : Float magnitude offest used for lightcurve compositing. Based on JPL Horizons magnitude differences.
    - shuffled_mags : If True, shuffles magnitude data for use in simulations. 

    Returns:
    - Dataframe of magnitude, phase, and date data. Binned data if binning=True.
    """

    # Observatory keys for date separation
    T_lookup = {"k21": 365 + 15 + 30 + 31 + 7, "k20": 8, "pmo": 0}
    T = T_lookup.get(observatory, 0)

    # Compute fractional days and phase zero point
    fractional_days = convert_utc_to_fractional_day(utc_list, offset, T)
    
    # Compute target magnitudes
    raw_mags = compute_target_magnitudes(target_df, comp_df, lit_df, drop_indices, system, return_mean=True)

    # Create table
    df = pd.DataFrame({
        "Asteroid Single Mag": raw_mags,
        "Date": fractional_days
    })

    # Phase calculation
    rotation_days = period / 24
    df["Phase"] = (df["Date"] % rotation_days) / rotation_days
    df["bin"] = (df["Phase"] / bin_width).astype(int)

    # Optional SDSS g' conversion
    if observatory in ["k21", "k20"]:
        df["g Mag"] = convert_to_sdss_g(df["Asteroid Single Mag"])

    # Apply magnitude offsets
    if 'g Mag' in df.columns:
        df['g Mag'] = df['g Mag'] - mag_offset
    elif 'Asteroid Single Mag' in df.columns:
        df['Asteroid Single Mag'] = df["Asteroid Single Mag"] - mag_offset
    
    # Shuffle magnitude data before binning if needed for simulating
    if shuffled_mags==True and 'g Mag' in df.columns:
        df['g Mag'] = np.random.choice(df['g Mag'], len(df['g Mag']), replace=False)
    elif shuffled_mags==True and 'Asteroid Single Mag' in df.columns:
        df['Asteroid Single Mag'] = np.random.choice(df['Asteroid Single Mag'], len(df['Asteroid Single Mag']), replace=False)

    # Binning
    n_bins = int(1.0 / bin_width)
    group_cols = {
        "sum_mag": ("Asteroid Single Mag", "sum"),
        "count": ("Asteroid Single Mag", "size"),
        "phase": ("Phase", "mean")
    }
    if "g Mag" in df.columns:
        group_cols["g_mag"] = ("g Mag", "sum")
    
    binned = (
        df.groupby("bin")
        .agg(**group_cols)
        .reindex(range(n_bins), fill_value=0)
        .reset_index()  # <-- This is key
    )
    binned = binned.reindex(range(n_bins), fill_value=0)
    binned["Asteroid Mean Mag"] = binned["sum_mag"] / binned["count"]
    binned["Asteroid Mean Mag"] = binned['Asteroid Mean Mag'].fillna(0)

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

def count_table_trial(params_list, period):
    """
    Function that folds data from phase_lightcurve calls and produces count data for a period trial.

    Parameters:
    - params_list : List of dictionaries of phase_lightcurve parameters for each observation.
    - period : Float period estimation for phasing.

    Returns:
    - List of counts for a given period trial.
    """

    # Dataframe initialization
    n_bins = int(1.0 / params_list[0]['bin_width'])  # assume consistent bin_width
    total_counts = pd.DataFrame({"count": [0.0] * n_bins})

    # Fold binned data and count elements in each phase bin
    for param in params_list:
        param = param.copy()  # Avoid modifying original
        param['period'] = period
        param['binning'] = True  # Ensure binning is active

        folded = phase_lightcurve(
            param['target_df'], param["comp_df"], param['lit_df'], param['utc_list'],
            **get_phase_params(param)
        )
        if "count" in folded.columns:
            total_counts["count"] += folded["count"].fillna(0)

    return total_counts

def scan_count_periods(params_list, start, stop, dp, save_path=None):
    """
    Function that iterates through trial periods and produces arrays for counts found in each phase bin.

    Paramters:
    - params_list : List of dictionaries of parameter inputs for phase_lightcurve.
    - start, stop : Float range of periods to sample over.
    - dp : Float step size to scan period range.
    - save_path : If True, saves the results to a .npz file for easier access.

    Returns:
      - Full count table (bins * trial)
      - Average counts per bin array
      - Total counts per bin array
    """

    # Initialize
    trial_period = start
    trial_index = 0
    count_tbl = pd.DataFrame()

    count_list = []
    period_labels = []

    trial_period = start
    trial_index = 1

    # Loop over sample range and construct dataframe of count results
    while trial_period <= stop:
        trial_counts = count_table_trial(params_list,trial_period)
        count_list.append(trial_counts["count"].reset_index(drop=True))
        period_labels.append(f"period_{trial_index}")
        
        trial_period += dp
        trial_index += 1

    # Combine all count columns at once
    count_tbl = pd.concat(count_list, axis=1)
    count_tbl.columns = period_labels

    # Compute average and total counts per bin
    average_counts = count_tbl.mean(axis=1).values
    total_counts = count_tbl.sum(axis=1).values

    # Save results as an npz for quicker function processing in latter steps
    if save_path is not None:
        np.savez(save_path, count_tbl=count_tbl, avg_counts=average_counts, total_counts=total_counts)

    return count_tbl, average_counts, total_counts

def calculate_sigma(
    period,
    params_list,
    avg_counts,
    total_counts,
    return_table=False,
    return_resid=False,
    shuffled_mags=False
):
    """
    Calculate the normalized sigma of a light curve phase-folded at a given period.
    Using weighted binning based on count_tbl outputs.

    Parameters:
    - period : Float trial period to test.
    - params_list : List of parameter dicts, one per dataset, for use with phase_lightcurve.
    - avg_counts : Array of mean counts per bin (averaged across datasets).
    - total_counts : Array of the sum of all individual bin counts across all datasets.
    - return_table : If True, return the combined bin table instead of sigma.
    - return_resid : If True, return squared residual instead of sigma.

    Returns:
    - float or DataFrame: sigma, squared residual, or the combined bin table.
    """

    n_bins = len(avg_counts)
    bin_width = 1.0 / n_bins

    # Initialize cumulative bin table
    cbin = pd.DataFrame({
        "sum_mag": [0.0] * n_bins,
        "count": [0.0] * n_bins,
        "phase": [0.0] * n_bins,
        "sum_phase": [0.0] * n_bins,
    })

    # Fold each dataset using the provided parameter set
    for param_dict in params_list:
        param_dict = param_dict.copy()  # Avoid modifying original
        param_dict.update({
            "period": period,
            "binning": False,
        })

        # Shuffle magnitudes if simulating
        if shuffled_mags==True:
            param_dict.update({"shuffled_mags": True})

        df = phase_lightcurve(
            param_dict['target_df'], param_dict["comp_df"], param_dict['lit_df'], param_dict['utc_list'],
            **get_phase_params(param_dict)
        )

        df["bin"] = (df["Phase"] / bin_width).astype(int)
        df = df[df["bin"] < n_bins]  # safety for edge bins

        # Determine magnitude column
        if "g Mag" in df.columns:
            df["mag"] = df["g Mag"]
        else:
            df["mag"] = df["Asteroid Single Mag"]

        # Group raw data into bins
        binned_df = df.groupby("bin").agg(
            sum_mag=("mag", "sum"),
            count=("mag", "count"),
            sum_phase=("Phase", "sum")
        ).reindex(range(n_bins), fill_value=0)

        cbin["sum_mag"] += binned_df["sum_mag"]
        cbin["count"] += binned_df["count"]
        cbin["sum_phase"] += binned_df["sum_phase"]

    # Average across datasets
    cbin["mean_phase"] = cbin["sum_phase"] / cbin["count"]
    cbin["mean_mag"] = cbin["sum_mag"] / cbin["count"]
    cbin["mean_mag"] = cbin["mean_mag"].fillna(0)
    cbin.drop(columns=["phase"], inplace=True)

    valid_bins = cbin["count"] > 0
    xmean = cbin.loc[valid_bins, "mean_mag"].mean()

    # Weighting
    norm_avg_counts = avg_counts / np.sum(avg_counts)
    weights = pd.Series(norm_avg_counts / np.sqrt(np.sum(total_counts)), index=cbin.index)

    # Calculate weighted residuals and sigma
    residuals = (cbin.loc[valid_bins, "mean_mag"] - xmean)
    weighted_residuals = residuals * weights[valid_bins]
    W_tot = weights[valid_bins].sum()

    normalized_resid = weighted_residuals / W_tot
    squared_resid = (normalized_resid ** 2).sum()
    normalized_sigma = np.sqrt((normalized_resid ** 2).mean())

    if return_table:
        return cbin
    elif return_resid:
        return squared_resid
    else:
        return normalized_sigma
    
def scan_sigma_periods(
    trial_periods,
    params_list,
    avg_counts,
    total_counts,
    verbose=False,
    return_best=False
):
    """
    Scan a list of trial periods to compute the normalized sigma for each,
    using weighted binning based on count_tbl outputs.

    Parameters:
    - trial_periods : List/array of trial periods to test.
    - params_list : List of parameter dicts (one per dataset) for phase_lightcurve.
    - avg_counts : Array of averaged bin counts (from count_tbl).
    - total_counts : Array of the sum of all bin counts across all datasets.
    - verbose : If True, prints progress.
    - return_best : If True, also returns the best period and min sigma.

    Returns:
    - pd.DataFrame: Columns = ['period', 'sigma']
    - (optional) float: Best-fit period
    - (optional) float: Minimum sigma
    """

    results = []
    
    for i, trial_period in enumerate(trial_periods):
        if verbose and i % 10 == 0:
            print(f"[INFO] Scanning period {i+1}/{len(trial_periods)}: {trial_period:.6f}")
        
        sigma = calculate_sigma(
            period=trial_period,
            params_list=params_list,
            avg_counts=avg_counts,
            total_counts=total_counts,
            shuffled_mags=False
        )
        results.append((trial_period, sigma))
    
    df_results = pd.DataFrame(results, columns=["period", "sigma"])

    if return_best:
        max_row = df_results.loc[df_results["sigma"].idxmax()]
        return df_results, max_row["period"], max_row["sigma"]
    
    return df_results