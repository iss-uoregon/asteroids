import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons
from src.data_loader import load_dataset, load_all_datasets
from astropy.time import Time  # For fallback epoch if needed
# from src.magnitude import phase_lightcurve

def get_dataset_params(config, dataset_name, period, offset=None,
                       bin_width=0.05, drop_indices=None, system="generic",
                       observatory="generic", mags=False, show=False, binning=False, mag_offset=None, shuffled_mags=False):
    """
    Given a base dataset name (e.g., 'pmo'), return the ordered parameters
    for phase_lightcurve(), with UTC column converted to a list and optional
    light-time correction automatically computed.

    Parameters:
    - config : yaml file of dataset specifications.
    - dataset_name : String reference to config file header used for each night's datasets.
    - period : Float period estimate for phasing.
    - offset : Float magnitude offset from light travel time.
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
    - Dictionary of dataset parameters for phase_lightcurve and related functions.
    """

    def load_dataset_by_name(name):
        d = next(d for d in config["data"]["datasets"] if d["name"] == name)
        return load_dataset(d)

    # Load relevant datasets
    target_df = load_dataset_by_name(f"{dataset_name}_target")
    comp_df   = load_dataset_by_name(f"{dataset_name}_comp")
    lit_df    = load_dataset_by_name(f"{dataset_name}_gaia")
    utc_df    = load_dataset_by_name(f"{dataset_name}_utc")

    # Convert UTC column to string list
    utc_list = utc_df["UTC"].dropna().astype(str).tolist()

    # Compute offset if not provided and light_time data exists in config
    config_entry = next(d for d in config["data"]["datasets"] if d["name"] == f"{dataset_name}_target")
    if offset is None and "light_time" in config_entry:
        target_tag = config_entry["light_time"]["target_tag"]
        try:
            epoch = config_entry["light_time"]["epoch"]
        except KeyError:
            raise KeyError(f"No 'epoch' defined in light_time section of config for dataset '{dataset_name}'. "
                        "Please add one explicitly to the config.")
        offset = compute_light_time_offset(target_tag, epoch)

    # Return ordered arguments ready for phase_lightcurve()
    return {
        "target_df": target_df,
        "comp_df": comp_df,
        "lit_df": lit_df,
        "utc_list": utc_list,
        "period": period,
        "offset": offset,
        "bin_width": bin_width,
        "drop_indices": drop_indices,
        "system": system,
        "observatory": observatory,
        "mags": mags,
        "show": show,
        "binning": binning,
        "mag_offset": mag_offset,
        "shuffled_mags": shuffled_mags
    }

def transform_magnitude_system(G, Gbp, Grp, system='johnson'):
    """Convert Gaia DR2 bands to synthetic magnitude using a specified system."""
    if system == 'sdss':
        return -0.13518 + G + 0.46245 * (Gbp - Grp) + 0.25171 * (Gbp - Grp)**2 - 0.021349 * (Gbp - Grp)**3
    elif system == 'johnson':
        return 0.01760 + G + 0.006860 * (Gbp - Grp) + 0.1732 * (Gbp - Grp)**2
    else:
        raise ValueError(f"Unknown photometric system: {system}")
    
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def convert_utc_to_fractional_day(utc_list, offset=0, T_offset=0):
    """
    Convert a list or Series of UTC strings (HH:MM:SS) to fractional day,
    applying optional offset and time shift.
    
    Parameters:
    - utc_list: List or pd.Series of time strings ("hh:mm:ss")
    - offset: Float, value to subtract from the fractional day
    - T_offset: Float, additional constant to add to the result
    
    Returns:
    - list of fractional day values with offset applied
    """

    # Convert to Series if needed
    utc_series = pd.Series(utc_list)

    # Drop malformed entries
    valid_times = utc_series[utc_series.str.count(":") == 2]

    # Split and convert
    try:
        hms = valid_times.str.split(":", expand=True).astype(float)
        hours, minutes, seconds = hms[0], hms[1], hms[2]

        total_seconds = hours * 3600 + minutes * 60 + seconds
        fractional_day = total_seconds / 86400.0
        result = (T_offset + fractional_day - offset).tolist()
    except Exception as e:
        print(f"[ERROR] Failed to convert UTC strings: {e}")
        result = []

    return result

def compute_light_time_offset(target_tag: str, reference_jd: float) -> float:
    """
    Compute the light-time correction (offset) in days for a target object
    observed from Earth at a given Julian Date.

    Parameters:
    - target_tag : String name or designation of the solar system object (e.g., 'Sabine').
    - reference_jd : Float Julian Date of the observation epoch.

    Returns:
    - Float Light-time correction in days.
    """

    obj = Horizons(id=target_tag, location='geo', epochs=reference_jd)
    obs = obj.ephemerides()

    distance_au = obs['delta'][0]
    distance_km = distance_au * 149597870.7  # AU to km
    light_time_sec = distance_km / 299792.458  # km/s (speed of light)
    light_time_days = light_time_sec / 86400  # seconds to days

    return light_time_days

def convert_to_sdss_g(mags):
    """This function will need to be updated to account for non-static B-V results, or relevant transformations"""
    return mags + 0.56 * 0.69 - 0.12

def get_phase_params(params):
    """Function used for specifying parameters within the get_dataset_params function for use in phase_lightcurve"""
    allowed = {
        "period", "offset", "drop_indices",
        "system", "observatory", "binning", "mag_offset", "shuffled_mags"
    }
    return {k: v for k, v in params.items() if k in allowed}

def get_err_params(params):
    """Function used for specifying parameters within the get_dataset_params function for 
    use in calculate_uncertainty_from_comparisons"""
    allowed = {
        "system", "drop_indices" 
    }
    return {k: v for k, v in params.items() if k in allowed}

def get_target_params(params):
    """Function used for specifying parameters within the get_dataset_params function for 
    use in compute_target_magnitudes"""
    allowed = {
        "system", "drop_indices" 
    }
    return {k: v for k, v in params.items() if k in allowed}

def plot_composite_lightcurve(
    params_list,
    phase_fct,
    period,
    period_err,
    terr_list,
    labels,
    colors,
    markers,
    amp_str,
    jd0_str,
    fig_label="Figure",
    fig_num=1,
    title="Phased Light Curves",
    err_frac=[0.05],
    figsize=(8, 5),
    ylim=(12.7, 13.35),
    xlim=(-0.05, 1.05),
    save_path=None,
    show=True
):
    """
    Plot composite phased light curves from multiple datasets.

    Parameters:
    - params_list: List of parameters for phase_lightcurve.
    - phase_fct: Function that produces phased dataframes.
    - period: Float phasing period.
    - period_err: Float phasing period error.
    - terr_list: List of y-error arrays, one for each dataset.
    - mag_offsets: List of magnitude offsets to apply.
    - labels: List of labels for each dataset.
    - colors: List of color hex codes.
    - markers: List of marker styles.
    - amp_str, jd0_str: Strings to annotate above plot.
    - fig_label: e.g., "Figure".
    - fig_num: Int number to display as "Figure N".
    - err_frac: List of fraction of points to display with errorbars per lightcurve.
    """

    fig, ax = plt.subplots(figsize=figsize)

    for param, yerr, label, color, marker, err in zip(
        params_list, terr_list, labels, colors, markers, err_frac
    ):
    
        param = param.copy()  # Avoid modifying original
        param['period'] = period
        param['binning'] = False  # Ensure binning is disabled

        df = phase_fct(
            param['target_df'], param["comp_df"], param['lit_df'], param['utc_list'],
            **get_phase_params(param)
        )

        phase = df["Phase"]
        if 'g Mag' in df.columns:
            mag = df['g Mag']
        else:
            mag = df['Asteroid Single Mag']
        ax.scatter(phase, mag, s=7.5, c=color, marker=marker, label=label)

        # Error bar thinning
        step = max(1, int(len(df) * err))
        idx = np.linspace(0, len(df) - 1, step).astype(int)
        ax.errorbar(
            phase.iloc[idx],
            mag.iloc[idx],
            yerr=yerr,
            fmt='.',
            ecolor=color,
            mec=color,
            mfc=color,
            markersize=5 if marker == 'o' else 2.5,
            capsize=3
        )

    # Inline top text annotations
    ax.text(0.11, ylim[0] + 0.03, jd0_str)
    ax.text(0.4,  ylim[0] + 0.03, f"Period: {period} ± {period_err}")
    ax.text(0.78, ylim[0] + 0.03, f"Amp: {amp_str}")

    # Figure label (left-aligned)
    ax.text(xlim[0], ylim[0] - 0.015, f"{fig_label} {fig_num}", fontsize=12, fontweight='bold')

    ax.set_xlabel('Phase')
    ax.set_ylabel("Relative Magnitude (g')")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc='lower left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show: plt.show()