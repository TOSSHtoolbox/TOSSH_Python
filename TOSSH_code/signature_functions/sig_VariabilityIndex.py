import numpy as np
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck


def sig_VariabilityIndex(Q, t):
    """
    Calculates variability index (VI) from FDC.
    VI is the standard deviation of the common logarithms of discharge
    determined at 10% intervals from 10% to 90% of the cumulative frequency
    distribution (flow duration curve, FDC). Low variability index shows
    higher water storage (Estrany et al., 2010).

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)

    Returns:
        tuple: (VariabilityIndex, error_flag, error_str)
        VariabilityIndex: variability index [-]

    Example:
        VariabilityIndex,_,_ = sig_VariabilityIndex(Q, t)

    References:
    Estrany, J., Garcia, C. and Batalla, R.J., 2010. Hydrological response
    of a small mediterranean agricultural catchment. Journal of Hydrology,
    380(1-2), pp.180-190.
    """

    # ensure consistent formatting
    Q = np.asarray(Q, dtype=float)

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q, t)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # calculate signature
    # get ranks as a proxy for exceedance probabilities
    Q_tmp = Q[~np.isnan(Q)]  # remove NaN values
    Q_sorted = np.sort(Q_tmp)[::-1]

    # percentiles required are 10%, 20%, ..., 90%
    percs = np.arange(10, 100, 10)

    # get the corresponding rank of the FDC values
    # (round half away from zero as in Matlab, np.round rounds half to even)
    ranks_percs = np.floor(len(Q_sorted) * percs / 100 + 0.5).astype(int)

    # too few values to define the 10th percentile (rank 0)
    if ranks_percs[0] < 1:
        error_flag = 3
        error_str = "Error: Not enough flow values to calculate variability index. " + error_str
        return np.nan, error_flag, error_str

    # get the flow value at each rank (1-based rank -> 0-based index)
    flow_percs = Q_sorted[ranks_percs - 1]

    # variation needed if some flow percentiles are zero for an intermittent
    # stream - exclude these from the calculation
    recs = flow_percs > 0

    # add warning for intermittent streams
    if np.any(Q_tmp == 0):
        error_flag = 2
        error_str = "Warning: Zero flows excluded from calculation. " + error_str

    # VI is the standard deviation of the common logarithms of discharge
    # determined at 10% intervals from 10% to 90% of the cumulative frequency
    # distribution (ddof=1 to match Matlab's std)
    VariabilityIndex = np.std(np.log10(flow_percs[recs]), ddof=1)

    return VariabilityIndex, error_flag, error_str
