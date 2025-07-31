import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, root_scalar
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from src.utils import transform_magnitude_system, gaussian, get_err_params, get_target_params
from src.magnitude import compute_comparison_magnitudes, compute_target_magnitudes, calculate_sigma
    
def calculate_uncertainty_from_comparisons(
    lit_df,
    comp_df,
    system='johnson',
    drop_indices=None,
    constrain_range=None,
    show=False,
    return_outliers=False
):
    """
    Estimate uncertainty in comparison star photometry by analyzing magnitude differences.
    
    Parameters:
        lit_df : Dataframe of literature magnitudes, should contain Gaia columns.
        comp_df (pd.DataFrame): Dataframe of measured field star photometrics.
        system : String of used photometric system 'sdss', or 'johnson'
        drop_indices : Optional list of star indices to exclude.
        constrain_range : Optional range (min, max) to filter mag diffs.
        show : If True, show histogram and Gaussian fit.
        return_outliers : If True, return list of flagged outliers instead of sigma.
        
    Returns:
        sigma_fit (float) or list of outlier strings if return_outliers is True.
    """

    df_lit = lit_df.filter(regex='phot_.*_mean_mag').copy()
    df_meas = comp_df.filter(regex='Source-Sky').copy()

    # Overly complicated drop indices statements from bug testing
    if drop_indices:
        valid_positions_lit = [i for i in drop_indices if i < len(df_lit)]
        valid_positions_meas = [i for i in drop_indices if i < df_meas.shape[1]]

        if valid_positions_lit:
            idx_to_drop = df_lit.index[valid_positions_lit]
            df_lit = df_lit.drop(index=idx_to_drop).reset_index(drop=True)
        else:
            print("[INFO] No valid indices to drop from df_lit.")

        if valid_positions_meas:
            df_meas = df_meas.drop(columns=df_meas.columns[valid_positions_meas])
        else:
            print("[INFO] No valid indices to drop from df_meas.")

    # Transform literature magnitudes into specified photometric system
    G, Gbp, Grp = df_lit.iloc[:, 0], df_lit.iloc[:, 1], df_lit.iloc[:, 2]
    Mag = transform_magnitude_system(G, Gbp, Grp, system=system)

    if df_meas.shape[1] > len(Mag):
        df_meas = df_meas.iloc[:, :len(Mag)]
    elif df_meas.shape[1] < len(Mag):
        Mag = Mag[:df_meas.shape[1]]

    Mag = pd.Series(Mag).reset_index(drop=True)

    # Form dictionaries of comparison star magnitude comparisons
    comparison_dict = {}

    for i, Ma in enumerate(Mag):
        Ia = df_meas.iloc[:, i]
        for j in range(len(Mag)):
            if j != i:
                Mb = Ma + 2.5 * np.log10(Ia / df_meas.iloc[:, j])
                col_name = f"Star {i} v {j}"
                comparison_dict[col_name] = Mb

    # Create the comparisons DataFrame
    comparisons = pd.DataFrame(comparison_dict)

    # Form list of comparison star differences from their literature value
    results = []
    outlier_logs = []
    for star_idx in range(len(Mag)):
        lit_mag = Mag[star_idx]
        star_comps = comparisons.filter(regex=f'v {star_idx}')
        for obs_idx in range(len(star_comps)):
            diffs = star_comps.iloc[obs_idx, :].values - lit_mag
            for diff in diffs:
                if abs(diff) > 0.4:
                    outlier_logs.append(f'Field {star_idx} obs {obs_idx} diff: {diff:.3f}')
            results.extend(diffs)

    if constrain_range:
        min_val, max_val = constrain_range
        results = [r for r in results if min_val <= r <= max_val]

    hist, bin_edges = np.histogram(results, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A_init = np.max(hist)
    mu_init = np.mean(results)
    sigma_init = np.std(results)

    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[A_init, mu_init, sigma_init])
    A_fit, mu_fit, sigma_fit = popt

    if show:
        x_fit = np.linspace(min(results), max(results), 1000)
        y_fit = gaussian(x_fit, *popt)
        plt.figure(figsize=(8, 5))
        plt.hist(results, bins=40, density=True, alpha=0.6, label="Magnitude Differences")
        plt.plot(x_fit, y_fit, 'r', label=f'Gaussian Fit\nσ = {sigma_fit:.4f}')
        plt.xlabel("Mag Difference from Literature")
        plt.ylabel("Density")
        plt.title("Magnitude Uncertainty Estimation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return outlier_logs if return_outliers else sigma_fit
    


def calculate_total_magnitude_uncertainty(
    dataset,
    transform_errors: list = [],
    verbose: bool = False
) -> float:
    """
    Computes the total magnitude uncertainty from multiple sources:
    - comparison star error
    - photometric measurement error (from target mag uncertainty column)
    - photometric system transformation errors (input manually)

    Parameters:
    - dataset : Keyword parameter for use in parameter fetching functions.
    - transform_errors : List of system transformation uncertainties (e.g., [0.045858, 0.03])
    - verbose : if True, print intermediate terms

    Returns:
    - float: total magnitude uncertainty
    """

    # Term 1: Comparison star error
    term1 = calculate_uncertainty_from_comparisons(
        dataset["lit_df"], dataset['comp_df'],
        **get_err_params(dataset),
        constrain_range=None,
        show=False,
        return_outliers=False
    )

    term1 = term1 ** 2

    # Term 2: Mean photometric measurement uncertainty
    target_df = compute_target_magnitudes(
        dataset['target_df'],dataset['comp_df'],dataset['lit_df'],
       **get_target_params(dataset), return_mean=False
    )

    term2 = target_df['Mag Uncertainty'].mean() ** 2

    # Term 3+: Transformation uncertainties, divided by sqrt(n_comps)
    n_comps = len([col for col in target_df.columns if col.startswith("Sabine Mag v star")])
    transform_terms = [(err / np.sqrt(n_comps)) ** 2 for err in transform_errors]

    total_variance = np.float64(term1) + np.float64(term2) + np.float64(sum(transform_terms))
    total_uncertainty = np.sqrt(total_variance)

    if verbose:
        print(f"Comparison Error^2: {term1:.6f}")
        print(f"Mean Photometric Error^2: {term2:.6f}")
        # print(f"Transform Terms: {transform_terms}")
        for i, t in enumerate(transform_terms):
            print(f"Transform Error {i+1}^2: {t:.6f}")
        print(f"Total Uncertainty: {total_uncertainty:.6f}")

    return total_uncertainty

def estimate_sigma_distribution(
    period_guess,
    calculate_sigma,
    params_list,
    avg_counts,
    total_counts,
    trials=5000,
    return_best_table=False,
    plot=True,
    bins=40,
    title=None,
    verbose=False
):
    """
    Run repeated trials of shuffled magnitude sigma calculation to estimate error distribution.

    Parameters:
    - period_guess : Float period to test sigma error around.
    - calculate_sigma_func : calculate_sigma function call.
    - params_list : List of dictionaries of parameter calls for phase_lightcurve function.
    - avg_counts : List of average binned counts for calculate_sigma normalization.
    - total_counts : List of total binned counts for calculate_sigma normalization.
    - plot : If True, generates histogram + Gaussian fit plot.
    - bins : Int number of bins for the histogram.
    - title : String plot title. If None, defaults to "Sigma Distribution at Period: {period_guess}"

    Returns:
    - mu_fit : Float mean of fitted Gaussian.
    - sigma_fit : Float standard deviation of fitted Gaussian.
    - fwhm : Float full width at half max of fitted Gaussian.
    - sigmas : list of floats of all calculated sigma values.
    - best_tbl : DataFrame or None of the Table with largest sigma if `return_best_table` is True, otherwise None.
    """

    # Initializing
    sigmas = []
    best_tbl = None
    max_sigma = float("-inf")

    # Loop over sampling range and collect the sigmas produced over the spread
    for trial in range(1, trials + 1):
        sigma = calculate_sigma(period_guess, params_list, avg_counts, total_counts, shuffled_mags=True)
        if verbose and trial % 10 == 0:
            print(f"[INFO] Scanning trial {trial}/{trials}: {sigma:.6f}")
        sigmas.append(sigma)

        if sigma > max_sigma:
            max_sigma = sigma
            if return_best_table:
                best_tbl = calculate_sigma(period_guess, params_list, avg_counts, 
                                           total_counts, return_table=True, shuffled_mags=True)

    # Histogram + Gaussian Fit
    hist, bin_edges = np.histogram(sigmas, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    A_init = np.max(hist)
    mu_init = np.mean(sigmas)
    sigma_init = np.std(sigmas)

    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[A_init, mu_init, sigma_init], maxfev=5000)
    A_fit, mu_fit, sigma_fit = popt
    fwhm = 2.355 * sigma_fit

    if plot:
        x_fit = np.linspace(min(sigmas), max(sigmas), 1000)
        y_fit = gaussian(x_fit, *popt)

        plt.figure(figsize=(8, 5))
        plt.hist(sigmas, bins=bins, density=True, alpha=0.6, label="Shuffled Sigma Histogram")
        plt.plot(x_fit, y_fit, color='red', label=f'Gaussian Fit\nσ: {sigma_fit:.5f}\nμ: {mu_fit:.5f}')
        plt.xlabel("Normalized Sigma")
        plt.ylabel("Probability Density")
        plt.title(title or f"Sigma Distribution at Period: {period_guess:.5f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(f"Fitted Mean Sigma (μ): {mu_fit:.5f}")
    print(f"Fitted Sigma Width (σ): {sigma_fit:.5f}")
    print(f"FWHM: {fwhm:.5f}")

    return mu_fit, sigma_fit, fwhm, sigmas, best_tbl if return_best_table else None

def fourier(phase, a0, a1, b1, a2, b2):
    """2nd-order Fourier Series"""
    return (a0
        + a1 * np.cos(2 * np.pi * phase)
        + b1 * np.sin(2 * np.pi * phase)
        + a2 * np.cos(4 * np.pi * phase)
        + b2 * np.sin(4 * np.pi * phase)
    )

def fourier_derivative(phase, a0, a1, b1, a2, b2):
    """Derivative of 2nd-order Fourier Series"""
    return (
        -2 * np.pi * a1 * np.sin(2 * np.pi * phase)
        + 2 * np.pi * b1 * np.cos(2 * np.pi * phase)
        -4 * np.pi * a2 * np.sin(4 * np.pi * phase)
        + 4 * np.pi * b2 * np.cos(4 * np.pi * phase)
    )

def fit_amplitude_models(x, y, period, mean_err, deg, smooth, filename=None, show=True):
    """
    Fitting function that produces polyfit, spline, and Fourier models over binned, composited lightcurve data.

    Parameters:
    - x : List of phase values from a calculate_sigma call at a given period.
    - y : List of asteroid magnitude values from a calculate_sigma call at a given period.
    - mean_err : Float averaged result from calculate_total_magnitude_uncertainty of each dataset.
    - deg : Int degree of the polyfit call.
    - smooth : Float smoothing factor for the Spline. A value of 0 directly fits to datapoints.
    - filename : If True, saves the resulting figure with the input string title.
    - show : If True, plots the three models over the binned, composite lightcurve.

    Returns:
     - Float amplitudes for the three models
     - Float roots for the three models
    """
    phase_grid = np.linspace(0, 1, 1000)
    errs = mean_err / np.sqrt(len(x))  # or use individual bin counts if available

    ### -- Polyfit --
    coeff = np.polyfit(x, y, deg=deg)
    poly_fct = np.poly1d(coeff)
    poly_fit = poly_fct(phase_grid)
    poly_deriv = poly_fct.deriv()
    poly_roots = poly_deriv.r
    poly_roots = poly_roots[(poly_roots >= 0) & (poly_roots <= 1)]
    poly_vals = poly_fct(poly_roots)
    amp_poly = max(poly_vals) - min(poly_vals)

    ### -- Spline --
    spline = UnivariateSpline(x, y, s=smooth, k=4)
    spline_fit = spline(phase_grid)
    spline_deriv = spline.derivative()
    spline_roots = spline_deriv.roots()
    spline_roots = spline_roots[(spline_roots >= 0) & (spline_roots <= 1)]
    spline_vals = spline(spline_roots)
    amp_spline = max(spline_vals) - min(spline_vals)

    ### -- Fourier --
    p0 = [np.mean(y), 0.1, 0.1, 0.05, 0.05]
    params, _ = curve_fit(fourier, x, y, p0=p0)
    fourier_fit = fourier(phase_grid, *params)

    def dF(phase): return fourier_derivative(phase, *params)
    def F(phase): return fourier(phase, *params)

    roots = []
    for start in np.linspace(0, 1, 500):
        end = start + 1e-2
        if end > 1:
            continue
        try:
            result = root_scalar(dF, bracket=[start, end], method='brentq')
            root = result.root
            if 0 <= root <= 1:
                roots.append(root)
        except (ValueError, RuntimeError):
            continue
    roots = np.unique(np.round(roots, 6))
    fourier_vals = F(np.array(roots))
    amp_fourier = max(fourier_vals) - min(fourier_vals)

    ### -- Plot --
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y, yerr=errs, fmt='.', ecolor='#1d61ad', capsize=3, alpha=0.7, label='Folded Lightcurve')
    plt.plot(phase_grid, spline_fit, c='#c72006', alpha=0.8, label='Spline Fit (s=0.003)')
    plt.plot(phase_grid, fourier_fit, c='#3ca135', ls='--', label='Fourier Fit (order=2)')
    plt.plot(phase_grid, poly_fit, c='#194e9e', ls='-.', label='Poly Fit (deg=6)')
    plt.xlabel("Phase")
    plt.ylabel("Relative Magnitude (g')")
    plt.title(f"Fitted Models to Light Curve (P={period:.5f}h)")
    plt.ylim(12.75, 13.33)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "amplitude_poly": amp_poly,
        "amplitude_spline": amp_spline,
        "amplitude_fourier": amp_fourier,
        "roots_fourier": roots,
        "roots_poly": poly_roots,
        "roots_spline": spline_roots
    }