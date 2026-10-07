import numpy as np
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck


def sig_x_percentile(Q,t,x):
    """
    Calculates the x-th flow percentile of streamflow.
    Following Addor et al. (2018) Q95 is a high flow measure, i.e. the 95%
    NON-exceedance probability.

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        x (array-like): x-th percentile(s) (e.g. 95 for Q95), can also be a numpy array 

    Returns:
    tuple: (x-th flow percentile, error_flag, error_str)

    Example: 
    Q_95,_,_ = sig_x_percentile(Q,t,95)

    References
    Addor, N., Nearing, G., Prieto, C., Newman, A.J., Le Vine, N. and
    Clark, M.P., 2018. A ranking of hydrological signatures based on their 
    predictability in space. Water Resources Research, 54(11), pp.8792-8812.
    """

    # ensure consistent formatting 
    Q = np.asarray(Q, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q,t)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # check range of input
    if np.any(x>100) or np.any(x<0):
        raise ValueError("x must be between 0 and 100")


    # calculate signature
    p = 1 - x/100  # exceedance probability

    # get ranks for exceedance probabilities
    Q_tmp = Q[~np.isnan(Q)]  # remove NaN values
    Q_sort = np.sort(Q_tmp)
    n = len(Q_tmp)
    Q_rank = np.arange(1, n + 1) # ranks 1..n
    FDC = 1 - Q_rank / n # calculate flow duration curve

    #initialize output
    Q_x = np.full(p.shape, np.nan)

    #find x-th flow percentile 
    for i, p_i in enumerate(p): 
        selected_ranks = Q_rank[FDC >= p_i]
        if selected_ranks.size > 0:                  # FDC may be ill-defined (e.g. x=0) -> NaN
            Q_x[i] = Q_sort[selected_ranks.max()-1] # 1-based rank -> 0-based index
            
    #add warning for intermittent streams
    if np.any(Q_tmp == 0):
        error_flag = 2
        error_str = "Warning: Flow is zero at least once (intermittent flow). " + error_str


    return Q_x, error_flag, error_str