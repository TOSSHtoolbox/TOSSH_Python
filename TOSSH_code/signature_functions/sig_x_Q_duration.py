import numpy as np
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck


def sig_x_Q_duration(Q, t, type, **kwargs):
    """
    sig_x_Q_duration calculates various kinds of flow durations.
    Calculates various kinds of flow durations, e.g. no flow duration, high flow duration.
    Typical metrics can be chosen from a standard list (no, high, low), or created manually
    (e.g. 0.5*median(Q)).

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        type (str): type of flow duration (no, high, low, custom_high, custom_low)
        threshold (float, optional): flow threshold above (below) which flow duration is calculated
            (e.g. 9*median(Q) for custom_high) [mm/timestep], required for custom_high and custom_low

    Returns:
        tuple: (x_Q_duration, error_flag, error_str)
        x_Q_duration: mean duration of the flow events [timestep]

    Example:
        x_Q_dur,_,_ = sig_x_Q_duration(Q, t, 'no')
        x_Q_dur,_,_ = sig_x_Q_duration(Q, t, 'high')
        x_Q_dur,_,_ = sig_x_Q_duration(Q, t, 'low')
        x_Q_dur,_,_ = sig_x_Q_duration(Q, t, 'custom_high', threshold=9*np.nanmedian(Q))

    References:
        Addor, N., Nearing, G., Prieto, C., Newman, A.J., Le Vine, N. and
        Clark, M.P., 2018. A ranking of hydrological signatures based on their
        predictability in space. Water Resources Research, 54(11), pp.8792-8812.
    """

    # Default values
    threshold = kwargs.get('threshold', None)

    # ensure consistent formatting
    Q = np.asarray(Q, dtype=float)
    threshold = np.array([]) if threshold is None else np.asarray(threshold, dtype=float).ravel()

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q,t)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # add warning for intermittent streams
    if np.any(Q == 0):
        error_flag = 2
        error_str = "Warning: Flow is zero at least once (intermittent flow). " + error_str

    # calculate signature
    # find timesteps with flow of the requested type (NaN compares as False, i.e. ends an event)
    if type == 'no':
        event_Q = Q == 0

    elif type == 'high':
        Q_high = 9 * np.nanmedian(Q)
        event_Q = Q > Q_high

    elif type == 'low':
        Q_low = 0.2 * np.nanmean(Q)
        event_Q = Q < Q_low

    elif type == 'custom_high':
        if threshold.size != 1:
            raise ValueError("No/wrong custom threshold specified")
        event_Q = Q > threshold[0]

    elif type == 'custom_low':
        if threshold.size != 1:
            raise ValueError("No/wrong custom threshold specified")
        event_Q = Q < threshold[0]

    else:
        raise ValueError("Incorrect flow duration type specified")

    # find consecutive timesteps of events
    padded = np.concatenate(([False], event_Q, [False])) # pad with False at both ends, same role as the 0 sentinels in MATLAB
    d = np.diff(padded.astype(np.int8))
    idx_starts = np.flatnonzero(d == 1)  # index of first True in each run
    idx_ends = np.flatnonzero(d == -1)   # one past the last True
    interval_lengths = idx_ends - idx_starts

    if type == 'high':
        if Q_high == 0:
            # add warning for intermittent streams
            interval_lengths = np.array([np.nan])
            error_flag = 2
            error_str = "Warning: median flow is 0 (intermittent flow), signature set to NaN. Consider alternative thresholds. " + error_str
        elif Q_high > np.nanmax(Q):
            error_flag = 2
            error_str = "Warning: 9 times median flow is larger than maximum flow (damped hydrograph). Consider alternative thresholds. " + error_str

    if interval_lengths.size == 0:
        x_Q_duration = 0
    else:
        x_Q_duration = np.mean(interval_lengths)

    return x_Q_duration, error_flag, error_str
