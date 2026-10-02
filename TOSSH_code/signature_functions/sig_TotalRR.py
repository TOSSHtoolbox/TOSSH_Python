import numpy as np
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck


def sig_TotalRR(Q,t,P):
    """
    Calculate total runoff ratio (TotalRR)

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        P (array-like): Precipitation [mm/timestep]
    
    Returns:
    tuple: (TotalRR, error_flag, error_str)

    Example
    TotalRR, error_flag, error_str = sig_TotalRR(Q,t,P)
    """

    # Initialize output variables
    error_flag = 0
    error_str = ""
    
    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q,t,P=P)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # calculate signature
    TotalRR = np.nanmean(Q)/np.nanmean(P)

    return TotalRR, error_flag, error_str