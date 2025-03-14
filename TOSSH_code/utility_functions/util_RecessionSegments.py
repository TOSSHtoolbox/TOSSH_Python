import numpy as np
import matplotlib.pyplot as plt
from utility_functions.util_LyneHollickFilter import util_LyneHollickFilter

# Function to perform recession segment identification
def util_RecessionSegments(
    Q,
    t,
    recession_length=5,
    n_start=0,
    eps=0,
    start_of_recession="peak",
    filter_par=0.925,
    # mask_thresh=np.nan,
    plot_results=False,
):
    """
    Identifies all individual recession segments.
    This function translated the `util_RecessionSegments` from the TOSSH toolbox in Matlab to Python.
    Source: https://github.com/TOSSHtoolbox/TOSSH/blob/master/TOSSH_code/utility_functions/util_RecessionSegments.m

    Parameters:
    Q : array-like
        Streamflow [mm/timestep]
    t : array-like
        Time [datetime or pandas datetime]
    recession_length : int, optional
        Minimum length of recession segments (days), default is 5 days. Renamed from "flow_section" in the original function
    n_start : int, optional
        Timesteps to be removed after start of recession, default is 1
    eps : float, optional
        Allowed increase in flow during recession period, default is 0
    start_of_recession : str, optional
        Defines start of recession when baseflow filter rejoins the curve ('baseflow') or after peak ('peak'), default is 'peak'
    filter_par : float, optional
        Smoothing parameter of Lyne-Hollick filter to determine start of recession (higher = later recession start), default is 0.925
    plot_results : bool, optional
        Whether to plot results, default is False

    Returns:
    recession_section : ndarray
        n-by-2 array where n is the number of recession segments and columns represent
        the indices of the start and end of recession segments.
    error_flag : int
        0 (no error), 1 (warning), 2 (error in data check), 3 (error in signature calculation)
    error_str : str
        String containing error description
    fig_handles : dict
        Dictionary containing figure handles if plot_results is True
    """

    # Initialize outputs
    fig_handles = {}
    error_flag = 0
    error_str = ""

    # Data check
    if np.isnan(Q).all():
        error_flag = 1
        error_str = "Recession was not extracted: All values in Q are NaN."
    else:
        if eps > np.nanmedian(Q) / 100:
            error_flag = 1
            error_str = "Warning: eps set to a value larger than 1 percent of median(Q). High eps values can lead to problematic recession selection."

    # Remove zero flow values
    is_zero = Q == 0
    Q_slice = Q.copy()
    Q_slice[is_zero] = np.nan
    Q = Q_slice.copy()

    # Calculate the minimum length of decreasing timesteps
    time_diff = t.iloc[1] - t.iloc[0]
    time_delta = time_diff.total_seconds() / 3600 / 24  # in days
    len_decrease = recession_length / time_delta

    # find timesteps with decreasing flow
    _decreasing_flow = Q[1:].to_numpy() < (
        Q[:-1].to_numpy() + eps
    )  # This returns logical array where Q(t+1)-Q(t) < 0 + eps

    # Find the index for the first decreasing flow
    if _decreasing_flow[0]:
        start_point = 0
    else:
        start_point = np.where(_decreasing_flow == 0)[0][0]

    # Find the decreasing flow section
    decreasing_flow = _decreasing_flow[start_point:]
    # If the flow record starts from decreasing values, change the first value of decreasing_flow to False, so that it gets detected by np.diff
    if _decreasing_flow[0]:
        decreasing_flow[0] = False

    # Find start and end of decreasing sections
    _flow_change = np.where(np.diff(decreasing_flow) != 0)[0] + start_point

    # If the flow is still decreasing at the end of the period, append that to flow change
    if decreasing_flow[-1]:
        _flow_change = np.append(_flow_change, len(Q))

    # Reshape into x by 2 array (columns = start, end of decrease)
    flow_change = _flow_change[: 2 * (len(_flow_change) // 2)].reshape(-1, 2)

    # Find segments that meet the recession length criteria
    recession_section = flow_change[
        (flow_change[:, 1] - flow_change[:, 0]) >= len_decrease + n_start
    ]
    recession_section[:, 0] += n_start # move start point n days

    Q_slice = Q.copy()
    Q_slice[is_zero] = 0 # do not use days with zero flow
    Q = Q_slice.copy()

    # Plot results if requested
    if plot_results:
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(t, Q, label="Streamflow", color="black")
        for section in recession_section:
            ax.plot(t[section[0] : section[1]], Q[section[0] : section[1]], color="red")
        ax.set_title("Recession Segments")
        fig_handles["RecessionSegments"] = fig
        # plt.show()

    # Baseflow adjustment if applicable
    # TODO: this "baseflow" part is not debugged yet
    if start_of_recession == "baseflow":
        Q_b = util_LyneHollickFilter(Q, filter_parameter=filter_par)
        is_baseflow = Q_b == Q
        for i in range(recession_section.shape[0]):
            baseflow_start = np.where(
                is_baseflow[recession_section[i, 0] : recession_section[i, 1]]
            )[0]
            if baseflow_start.size > 0:
                recession_section[i, 0] += baseflow_start[0]
        recession_section = recession_section[
            recession_section[:, 1] >= recession_section[:, 0] + 3
        ]
        if recession_section.size == 0:
            return [], 3, "No valid baseflow recession segments.", fig_handles

    # Final error check
    if recession_section.size < 10:
        error_flag, error_str = (
            1,
            "Warning: Fewer than 10 recession segments extracted.",
        )

    return recession_section, error_flag, error_str, fig_handles