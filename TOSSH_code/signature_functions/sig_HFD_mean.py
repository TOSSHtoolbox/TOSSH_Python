import numpy as np
from datetime import datetime
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck


def sig_HFD_mean(Q, t, **kwargs):
    """
    sig_HFD_mean calculates the mean half flow date (day since start of water year on which
    the cumulative discharge exceeds half of the annual discharge sum)

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        start_water_year (int, optional): first month of water year, default = 10 (October)

    Returns:
        tuple: (HFD_mean, error_flag, error_str)
        HFD_mean: mean half flow date [day since start of water year]

    Example:
        HFD_mean,_,_ = sig_HFD_mean(Q, t)
        HFD_mean,_,_ = sig_HFD_mean(Q, t, start_water_year=1)

    References:
        Court, A., 1962. Measures of streamflow timing. Journal of Geophysical
        Research, 67(11), pp.4335-4339.
    """

    # Default values
    start_water_year = kwargs.get('start_water_year', 10)

    # check range of input
    if not (isinstance(start_water_year, (int, np.integer)) and 1 <= start_water_year <= 12):
        raise ValueError("start_water_year must be an integer between 1 and 12")

    # ensure consistent formatting
    Q = np.asarray(Q, dtype=float)

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q,t)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # adjust for timestep
    timestep_days = timestep.total_seconds() / 86400

    # calculate signature
    #get years
    parsed = [
        datetime.strptime(str(d)[:10], "%Y-%m-%d") # [:10] removes the time portion, for example: # '1999-01-01T00:00:00.000000000' -> '1999-01-01'
        for d in t
    ]

    year_arr  = np.array([d.year for d in parsed])
    month_arr = np.array([d.month for d in parsed])

    #year range (continuous, like MATLAB year_start:year_end, so years missing from the record are not skipped)
    year_list = np.arange(year_arr.min(), year_arr.max() + 1)

    #initialize
    n_years = len(year_list)-1
    HFD = np.full((n_years,1),np.nan)

    #extract years
    error_tmp = False

    for i,year in enumerate(year_list[1:]): # since we use water years, we always start in the "2nd year"
        mask = ((year_arr == year - 1) & (month_arr >= start_water_year)) | \
            ((year_arr == year)     & (month_arr <  start_water_year)) # mask water year
        Q_water_year = Q[mask]

        Q_half_sum = 0.5*Q_water_year.sum()
        Q_cumsum = np.cumsum(Q_water_year)

        HFD_aux = np.flatnonzero(Q_cumsum > Q_half_sum)

        # no index is found if the water year contains NaN values or no data (MATLAB raises an error here)
        if len(HFD_aux) > 0:
            HFD[i,0] = HFD_aux[0] + 1 # 0-based index -> 1-based day count
        else:
            error_tmp = True

    if error_tmp:
        error_flag = 1
        error_str = "Warning: Years containing NaN values are ignored. " + error_str

    #get mean half flow date
    HFD_mean = np.nanmean(HFD) * timestep_days

    return HFD_mean, error_flag, error_str
