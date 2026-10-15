import numpy as np
import matplotlib.pyplot as plt
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck

def sig_FDC(Q, t, **kwargs):
    """
    Calculates flow duration curve (FDC)

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        plot_results (bool, optional): Whether to plot results, Default is false-

    Returns:
        tuple: (FDC, Q_sorted, error_flag, error_str, fig_handle)
        FDC: exceedance probabilities [-]
        Q_sorted: sorted streamflow values [mm/timestep]

    Example:
        FDC,Q_sorted,_,_,fig = sig_FDC(Q, t, plot_results=True)
    """
    # Default values
    plot_results = kwargs.get('plot_results', False)

    # Initialiize output
    fig_handles = {}

    # ensure consistent formatting 
    Q = np.asarray(Q, dtype=float)

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q,t)
    if error_flag == 2:
       return np.full(Q.shape, np.nan), np.full(Q.shape, np.nan),error_flag, error_str, fig_handles

    if not isinstance(plot_results, bool):
        raise TypeError("plot_results must be a bool")


    # get ranks for exceedance probabilities
    Q_tmp = Q[~np.isnan(Q)]  # remove NaN values
    Q_sorted = np.sort(Q_tmp)
    n = len(Q_tmp)
    Q_rank = np.arange(1, n + 1) # ranks 1..n
    FDC = 1 - Q_rank / n # calculate flow duration curve

    # add warning for intermittent streams
    if np.any(Q_tmp == 0):
        error_flag = 2
        error_str = "Warning: Flow is zero at least once (intermittent flow). " + error_str


    # Optional plotting
    if plot_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(FDC, Q_sorted)
        ax.set_yscale('log')
        ax.set_xmargin(0)
        ax.set_xlabel('Exceedance probability [-]')
        ax.set_ylabel('Q [mm/timestep]')
        fig_handles['FDC'] = fig
        plt.show()    

    return FDC, Q_sorted, error_flag, error_str, fig_handles

