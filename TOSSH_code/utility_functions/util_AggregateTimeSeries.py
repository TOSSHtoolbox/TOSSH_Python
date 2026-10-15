import numpy as np


def util_AggregateTimeSeries(X, t, **kwargs):
    """
    Calculates annual and monthly sums of time series.
    Start of water year can be specified so that the sum is taken over a water year.
    This function translated the `util_AggregateTimeSeries` from the TOSSH toolbox in Matlab to Python.
    Source: https://github.com/TOSSHtoolbox/TOSSH/blob/master/TOSSH_code/utility_functions/util_AggregateTimeSeries.m

    Parameters:
    X : array-like
        Time series - should be continuous (e.g. Q, typically [mm/timestep])
    t : array-like
        Time [datetime or numpy datetime64]
    start_water_year : int, optional
        First month of water year, default is 1 (January)

    Returns:
    X_annual : ndarray
        Annual sums [mm/year]
    X_monthly : ndarray
        n-by-12 array of monthly sums [mm/month], columns ordered from the first month of the water year
    year_list : ndarray
        Corresponding years (water year corresponds to the year of its last day,
        e.g. water year starting 1 October 1999 is water year 2000)
    days_per_year : ndarray
        Number of non-NaN timesteps per year
    error_flag : int
        0 (no error), 1 (warning), 2 (error in data check), 3 (error in signature calculation)
    error_str : str
        String containing error description
    """

    # default values
    start_water_year = kwargs.get('start_water_year',1)


    X = np.asarray(X, dtype=float).flatten()

    if not isinstance(start_water_year, (int, np.integer)) or not 1 <= start_water_year <= 12:
        raise ValueError("start_water_year must be an integer between 1 and 12.")


    # default setting reads as good data
    error_flag = 0
    error_str = ""

    # get years and months
    t = np.asarray(t).astype("datetime64[D]")
    year_vec = t.astype("datetime64[Y]").astype(int) + 1970
    month_vec = t.astype("datetime64[M]").astype(int) % 12 + 1
    day_vec = (t - t.astype("datetime64[M]")).astype(int) + 1

    if start_water_year == 1:
        # calendar year, all months lie in the same calendar year
        year_offset = 0
    else:
        # water year always corresponds to the last day of the water year,
        # e.g. water year starting from 1 October 1999 is water year 2000
        year_offset = 1
    year_list = np.arange(year_vec.min() + year_offset, year_vec.max() + 1)

    # last month of the water year (December for calendar year)
    end_water_year = (start_water_year - 2) % 12 + 1

    if month_vec[0] != start_water_year and day_vec[0] != 1:
        error_flag = 1
        error_str = "Warning: Time series and water year do not match. Incomplete years possible. " + error_str
    elif month_vec[-1] != end_water_year and day_vec[-1] < 28:
        error_flag = 1
        error_str = "Warning: Time series and water year do not match. Incomplete years possible. " + error_str

    X_annual = np.full(len(year_list), np.nan)
    X_monthly = np.full((len(year_list), 12), np.nan)
    days_per_year = np.full(len(year_list), np.nan)

    # extract years and months
    for y, year in enumerate(year_list):
        X_water_year = np.concatenate((
            X[(year_vec == year - year_offset) & (month_vec >= start_water_year)],
            X[(year_vec == year) & (month_vec < start_water_year)],
        ))
        X_annual[y] = np.nansum(X_water_year)
        days_per_year[y] = np.sum(~np.isnan(X_water_year))
        # months from the "previous" calendar year (same year for calendar year)
        for m in range(start_water_year, 13):
            X_monthly[y, m - start_water_year] = np.nansum(X[(year_vec == year - year_offset) & (month_vec == m)])
        # months from the "current" calendar year
        for m in range(1, start_water_year):
            X_monthly[y, m + 12 - start_water_year] = np.nansum(X[(year_vec == year) & (month_vec == m)])

    return X_annual, X_monthly, year_list, days_per_year, error_flag, error_str
