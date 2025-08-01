from src.magnitude import phase_lightcurve
from src.error_functions import calculate_total_magnitude_uncertainty
from src.utils import plot_composite_lightcurve, get_phase_params
from src.params_list import pmo, kobe20, kobe21, params_list

"""
Driver file for demonstrating magnitude uncertainty calculations,
phasing light curves, and plotting composite light curves using
example datasets (Sabine: PMO, Kobe 2020/2021).

Steps:
1. Compute target magnitude uncertainties
2. Test phasing at a trial period
3. Plot a composite lightcurve to compare datasets
"""
def main():
    # 1. Example for magnitude error calculations using Sabine files
    pmo_terr = calculate_total_magnitude_uncertainty(
        dataset=pmo,
        transform_errors=[0.16497],
        verbose=False
    )

    k20_terr = calculate_total_magnitude_uncertainty(
        dataset=kobe20,
        transform_errors=[0.045858, 0.03],
        verbose=False
    )

    k21_terr = calculate_total_magnitude_uncertainty(
        dataset=kobe21,
        transform_errors=[0.045858, 0.03],
        verbose=False
    )
    # print(pmo_terr)

    terr_list = [pmo_terr, k20_terr, k21_terr]

    # 2. Example of phased dataset for Sabine data
    phase_example = phase_lightcurve(
        kobe20['target_df'], kobe20["comp_df"], kobe20['lit_df'], kobe20['utc_list'],
        **get_phase_params(kobe20)
            )
    # print(phase_example)

    # 3. Example plot of composited Sabine data
    plot = plot_composite_lightcurve(
        params_list=params_list,
        phase_fct=phase_lightcurve,
        period=4.29283,
        period_err=0.00018,
        terr_list=terr_list,
        labels=['PMO 20200816','NHAO 20200824','NHAO 20211107'],
        colors=['#1d61ad', '#c25925', '#18a835'],
        markers=['o', 's', '^'],
        amp_str="0.30",
        jd0_str="JDo(UTC): 2459077.5",
        fig_label="Figure",
        fig_num=1,
        title="665 Sabine Phased Light Curves",
        err_frac=[0.04, 0.04, 0.03],
        show=True
    )
    # print(plot)

if __name__ == "__main__":
    main()
