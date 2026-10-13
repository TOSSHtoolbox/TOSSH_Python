import numpy as np
from TOSSH_code.utility_functions.util_DataCheck import util_DataCheck
from TOSSH_code.utility_functions.util_AggregateTimeSeries import util_AggregateTimeSeries


def sig_QP_elasticity(Q, t, P, **kwargs):
    """
    sig_QP_elasticity calculates streamflow-precipitation elasticity, the sensitivity of a
    catchment's streamflow response to changes in precipitation at the annual time scale.
    Start of water year can be defined.

    Args:
        Q (array-like): Streamflow [mm/timestep]
        t (array-like): Time (datetime object)
        P (array-like): Precipitation [mm/timestep]
        start_water_year (int, optional): first month of water year, default = 10 (October)
        method (str, optional): method to calculate elasticity ('Sawicz', 'Sanka'), default = 'Sanka'

    Returns:
        tuple: (QP_elasticity, error_flag, error_str)
        QP_elasticity: streamflow-precipitation elasticity [-]

    Example:
        QP_elasticity,_,_ = sig_QP_elasticity(Q, t, P)
        QP_elasticity,_,_ = sig_QP_elasticity(Q, t, P, start_water_year=1, method='Sawicz')

    References:
        Sankarasubramanian, A., Vogel, R.M. and Limbrunner, J.F., 2001. Climate
        elasticity of streamflow in the United States. Water Resources
        Research, 37(6), pp.1771-1781.
        Sawicz, K., Wagener, T., Sivapalan, M., Troch, P.A. and Carrillo, G.,
        2011. Catchment classification: empirical analysis of hydrologic
        similarity based on catchment function in the eastern USA. Hydrology
        and Earth System Sciences, 15(9), pp.2895-2911.
    """

    # Default values
    start_water_year = kwargs.get('start_water_year', 10)
    method = kwargs.get('method', 'Sanka')

    # check range of input
    if not (isinstance(start_water_year, (int, np.integer)) and 1 <= start_water_year <= 12):
        raise ValueError("start_water_year must be an integer between 1 and 12")
    if not isinstance(method, str):
        raise ValueError("method must be a string")

    # ensure consistent formatting
    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)

    # Data checks
    error_flag, error_str, timestep, t = util_DataCheck(Q, t, P=P)
    if error_flag == 2:
        return np.nan, error_flag, error_str

    # calculate signature
    error_flag_tmp = error_flag # temporarily store error flag from data check
    error_str_tmp = error_str

    # aggregate time series to get annual sums
    Q_annual, _, _, days_per_year, error_flag, error_str = util_AggregateTimeSeries(
        Q, t, start_water_year=start_water_year
    )
    P_annual, _, _, days_per_year, error_flag, error_str = util_AggregateTimeSeries(
        P, t, start_water_year=start_water_year
    )
    if error_flag == 0:
        error_flag = error_flag_tmp
        error_str = error_str_tmp

    # calculate elasticity
    # use days_per_year to adjust for leap years and incomplete years
    match method:
        case 'Sanka':
            dQ = Q_annual/days_per_year - np.nanmean(Q_annual/days_per_year)
            dP = P_annual/days_per_year - np.nanmean(P_annual/days_per_year)
        case 'Sawicz':
            dQ = np.diff(Q_annual/days_per_year)
            dP = np.diff(P_annual/days_per_year)
        case _:
            raise ValueError("Please choose one of the available methods (Sawicz or Sanka).")

    QP_elasticity = np.nanmedian((dQ/dP)*(np.nanmean(P)/np.nanmean(Q)))

    return QP_elasticity, error_flag, error_str
