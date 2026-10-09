import numpy as np
import matplotlib.pyplot as plt
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck

def sig_FDC_slope(Q, t, **kwargs):
    """
    Calculates slope of flow duration curve (FDC).
    Calculates slope of FDC between two normalised streamflow percentiles
    (e.g. 33rd and 66th, see McMillan et al., 2017). Slope can be fitted in
    linear or in log space. Note that percentiles are defined as exceedance
    probabilities, i.e. the low-section corresponds to high values, e.g.
    [0.8 1.0].

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        slope_range (array-like, optional): Range of FDC [perc_lower, perc_upper] in which slope
            should be calculated, Default is [0.33, 0.66]
        fitLogSpace (bool, optional): Whether to fit slope in log space or in linear space, Default is True
        plot_results (bool, optional): Whether to plot results, Default is False

    Returns:
        tuple: (FDC_slope, error_flag, error_str, fig_handles)
        FDC_slope: slope of flow duration curve [-]

    Example:
        FDC_slope,_,_,_ = sig_FDC_slope(Q, t)
        FDC_slope,_,_,fig = sig_FDC_slope(Q, t, slope_range=[0.33, 0.66], fitLogSpace=False, plot_results=True)

    References:
    McMillan, H., Westerberg, I. and Branger, F., 2017. Five guidelines for
    selecting hydrological signatures. Hydrological Processes, 31(26),
    pp.4757-4761.
    """
    # Default values
    slope_range = kwargs.get('slope_range', [0.33, 0.66])
    fitLogSpace = kwargs.get('fitLogSpace', True)
    plot_results = kwargs.get('plot_results', False)

    # Initialize output
    fig_handles = {}

    # ensure consistent formatting
    Q = np.asarray(Q, dtype=float)
    slope_range = np.sort(np.asarray(slope_range, dtype=float).ravel())  # check order of percentiles

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q, t)
    if error_flag == 2:
        return np.nan, error_flag, error_str, fig_handles

    if slope_range.size != 2 or np.any(slope_range < 0) or np.any(slope_range > 1):
        raise ValueError("slope_range must contain two values between 0 and 1")
    if not isinstance(fitLogSpace, bool):
        raise TypeError("fitLogSpace must be a bool")
    if not isinstance(plot_results, bool):
        raise TypeError("plot_results must be a bool")


    # get ranks for exceedance probabilities
    Q_tmp = Q[~np.isnan(Q)]  # remove NaN values
    Q_sorted = np.sort(Q_tmp)
    n = len(Q_tmp)
    Q_rank = np.arange(1, n + 1) # ranks 1..n
    FDC = 1 - Q_rank / n # calculate flow duration curve
    Q_median = np.median(Q_tmp)

    # slope of FDC between upper and lower percentile (1-based rank -> 0-based index)
    # due to the way the FDC is calculated, the maximum exceedance probability will always be <1
    if slope_range[1] == 1:
        bound_up = 0
    else:
        bound_up = Q_rank[FDC >= slope_range[1]].max() - 1
    bound_low = Q_rank[FDC >= slope_range[0]].max() - 1

    # fit slope, either in linear or in log space (normalised by median to make it dimensionless)
    if fitLogSpace:
        FDC_slope = (np.log(Q_sorted[bound_up]/Q_median) - np.log(Q_sorted[bound_low]/Q_median)) / \
            (FDC[bound_up] - FDC[bound_low])
    else:
        FDC_slope = ((Q_sorted[bound_up]/Q_median) - (Q_sorted[bound_low]/Q_median)) / \
            (FDC[bound_up] - FDC[bound_low])

    # in case flow is very intermittent (e.g. 66th percentile is 0)
    if not np.isfinite(FDC_slope):
        error_flag = 3
        error_str = "Error: FDC slope could not be calculated, probably because flow is intermittent. " + error_str
        return np.nan, error_flag, error_str, fig_handles

    # add warning for intermittent streams
    if np.any(Q_tmp == 0):
        error_flag = 2
        error_str = "Warning: Flow is zero at least once (intermittent flow). " + error_str


    # Optional plotting
    if plot_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(FDC[bound_low], FDC[bound_up], 0.001)
        if fitLogSpace:
            ax.plot(FDC, np.log(Q_sorted/Q_median), label='FDC')
            c = np.log(Q_sorted[bound_low]/Q_median) - FDC_slope*FDC[bound_low]
            ax.set_ylabel('log(Q/median(Q)) [-]')
        else:
            ax.plot(FDC, Q_sorted/Q_median, label='FDC')
            c = Q_sorted[bound_low]/Q_median - FDC_slope*FDC[bound_low]
            ax.set_ylabel('Q/median(Q) [-]')
        ax.plot(x, FDC_slope*x + c, '--', linewidth=2, label='Fitted slope')
        ax.legend()
        ax.set_xlabel('Exceedance probability [-]')
        fig_handles['FDC_slope'] = fig
        plt.show()

    return FDC_slope, error_flag, error_str, fig_handles
