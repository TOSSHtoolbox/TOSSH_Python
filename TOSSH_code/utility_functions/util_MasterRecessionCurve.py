import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def get_indices(x_vals, segment, num_x):
    """A helper function to get the indices of the flow segment in the Q gridspace

    Args:
        x_vals (_type_): Numpy array of observed values (for example, Q values)
        segment (_type_): NumPy array of a recession segment to be interpolated
        num_x (_type_): Number of interpolated points in the MRC

    Return
        fmin_index: Index of the maximum value of the segment in the x grid space
        fmax_index: Index of the minimum value of the segment in the x grid space
        nf: Number of grids that the recession segment covered
    """
    # Calculate fmax_index
    if segment.iloc[0].item() >= x_vals.max():
        fmax_index = 0
    else:
        fmax_index = num_x - np.searchsorted(x_vals[::-1], segment.iloc[0], side="left")
        fmax_index = int(np.floor(fmax_index.item()))

    # Calculate fmin_index
    if segment.iloc[-1].item() <= x_vals.min():
        fmin_index = num_x - 1
    else:
        fmin_index = num_x - np.searchsorted(
            x_vals[::-1], segment.iloc[-1], side="left"
        )
        fmin_index = int(np.floor(fmin_index.item()))

    # Number of grids that the recession segment covered
    nf = fmin_index - fmax_index

    return fmin_index, fmax_index, nf


def util_MasterRecessionCurve(
    Q,
    flow_section,
    fit_method="exponential",
    match_method="log",
    plot_results=False,
):
    """
    Fits a master recession curve (MRC) to recession segments.
    This function translated the `util_MasterRecessionCurve` from the TOSSH toolbox in Matlab to Python, using ChatGPT + manual edit and debugs.
    Source: https://github.com/TOSSHtoolbox/TOSSH/blob/master/TOSSH_code/utility_functions/util_MasterRecessionCurve.m

    Parameters:
    Q : array-like
        Streamflow [mm/timestep] 
    flow_section : array-like
        n-by-2 array of start and end indices of recession segments;
        columns are the indices into the flow array of the start and end of the recession segments
    OPTIONAL:
    fit_method : str, optional
        'exponential' (approximates each recession segment as an
        exponential before stacking into MRC), 'nonparameteric' (fits 
        horizontal time shift for minimum standard deviation at each lag
        time, does not assume any form of the curve), default is 'exponential'
    match_method : str, optional
        how to space points on the MRC used for alignment, 'linear' or 'log', default is 'log'
    plot_results : bool, optional
        Whether to plot results, default is False

    Returns:
    MRC : array-like
        Two-column array of time and flow, specifying the MRC
    fig_handles : list
        Figure handles to manipulate figures (empty if plotting is not requested)

    EXAMPLE:
    # load example data
    Q = data.Q;
    t = data.t;
    flow_section = util_RecessionSegments(Q,t); % get recession segments
    [mrc] = util_MasterRecessionCurve(Q, flow_section); % get MRC

    References:
    - Posavec, K., Bacani, A. and Nakic, Z., 2006. A visual basic spreadsheet macro for recession curve analysis. Groundwater, 44(5), pp.764-767.
    - Gnann, S.J., Coxon, G., Woods, R.A., Howden, N.J.K., McMillan H.K., 2021. TOSSH: A Toolbox for Streamflow Signatures in Hydrology. Environmental Modelling & Software. https://doi.org/10.1016/j.envsoft.2021.104983
    """

    if fit_method == "exponential":
        # _________________________________________________________________________________
        # sort the flow sections according to highest starting value

        # Get the Q values corresponding to the start of the recession
        start_values = Q[
            flow_section[:, 0]
        ]  

        # sort the flow sections according to highest starting value
        sorted_indices = np.argsort(
            -start_values
        )  

        sorted_flow_section = flow_section[
            sorted_indices
        ]  
        
        # _________________________________________________________________________________
        # Start the MRC with the highest segment
        # MRC=[relative time, Q values]
        mrc_array = np.column_stack(
            (
                np.arange(
                    sorted_flow_section[0, 1] - sorted_flow_section[0, 0] + 1
                ),
                Q[sorted_flow_section[0, 0] : sorted_flow_section[0, 1] + 1],
            )
        )

        # _________________________________________________________________________________
        # Prepare the MRC plot
        if plot_results:
            fig, ax = plt.subplots(figsize=(8, 6))

        # _________________________________________________________________________________
        # Loop adding segment to MRC each time
        for i in range(1, sorted_flow_section.shape[0]):
            # Fit an exponential to the mrc so far lny=ax+b

            mask = ~np.isnan(mrc_array[:, 0]) & ~np.isnan(mrc_array[:, 1])

            # Apply polyfit only to valid (non-NaN) data:
            # Assume linear relationship between the x=relative time and y=log(x)
            # log(y) = a*x * b
            # a = mdl[1], and b = mdl[0]

            mdl = np.polyfit(x=mrc_array[mask, 0], y=np.log(mrc_array[mask, 1]), deg=1)

            # Get the flow section
            current_segment = Q[
                sorted_flow_section[i, 0] : sorted_flow_section[i, 1] + 1
            ]

            # Time shift for alignment:
            # Calculate the time shift rexuired to place the initial point
            # of the next recession segment on the first regression curve
            # time_shift = (np.log(x[sorted_flow_section[i, 1]]) - mdl[0]) / mdl[1]
            time_shift = (np.log(current_segment.iloc[0] - mdl[0])) / mdl[1]

            # Add the shifted segment to the MRC
            # new_time = relative time + time shift
            new_time = (
                np.arange(
                    sorted_flow_section[i, 1] - sorted_flow_section[i, 0] + 1
                )
                + time_shift
            )

            # Add the shifted segment to the master recession
            mrc_array = np.vstack(
                (mrc_array, np.column_stack((new_time, current_segment)))
            )

            # Optional plotting
            if plot_results:
                ax.plot(new_time, current_segment, "k", linewidth=1)

        if plot_results:
            ax.set_xlabel("Time after recession start")
            ax.set_ylabel("Data [mm/timestep]")
            ax.legend(["Data"])
            ax.set_title("Master Recession Curve")
            # fig.show()  # Display the figure (you could use plt.show() as well)

    elif fit_method == "nonparametric":
        # download all the flow segments, add jitter to avoid long constant
        # flow values that can't be interpolated, sort values to avoid
        # cases where flow is not decreasing, find min and max flow
    
        # Constants
        jitter_size = 1e-8  # jitter size to avoid constant flow values
        num_Q = 500  # number of interpolated points in the MRC

        # _________________________________________________________________________________
        numsegments = len(flow_section)  # get number of flow segments

        # Order flow sections starting with the largest initial flow value
        Q_init_value = Q[flow_section[:, 0]]
        sortind = np.argsort(Q_init_value)[::-1]
        running_min = Q_init_value.max()  # Keep running tally of minimum

        # Create list of recession segments, starting with highest flow
        # Add jitter to everything except the first value of each segment
        segments = []
        for i in range(numsegments):
            # Retrieve the recession segment x dta values
            segment = Q[
                flow_section[sortind.iloc[i], 0] : flow_section[
                    sortind.iloc[i], 1
                ]
                + 1
            ]

            # Add jitter
            segment[1:] += np.random.normal(0, jitter_size, len(segment) - 1)

            # TODO: Have not translated the following part of the Matlab code
            # % avoid negative segment values
            # segment = abs(segment)+1e-20;
            # % sort the segment with jitter, in case eps parameter was used
            # % and so thereare small increases during the recessions
            # segment = sort(segment,'descend');

            # Store
            segments.append(segment)

        # Get flow values where curves should be matched
        max_Q = max([np.max(seg) for seg in segments])
        min_Q = max(min([np.min(seg) for seg in segments]), jitter_size)

        # _________________________________________________________________________________
        # Get interpolated Q values where MRC will be evaluated

        # Get the x gridspace in log or linear space
        if match_method == "linear":
            Q_vals = np.linspace(max_Q, min_Q, num_Q)

        elif match_method == "log":
            frac_log = 0.2
            gridspace = (max_Q - min_Q) / num_Q
            linspace_part = np.linspace(
                max_Q - gridspace / 2,
                min_Q + gridspace / 2,
                num_Q - int(frac_log * num_Q),
            )
            logspace_part = np.logspace(
                np.log10(max_Q), np.log10(min_Q), int(frac_log * num_Q)
            )
            Q_vals = np.sort(np.unique(np.concatenate([linspace_part, logspace_part])))[
                ::-1
            ]
            Q_vals[-1] = min_Q
            Q_vals[0] = max_Q

        else:
            raise ValueError("Match method for MRC not a recognized option.")

        # _________________________________________________________________________________
        # Track good segments: Extract and interpolate each segment, check validity; remove invalid segments within the gridspace
        short_segs = np.zeros(numsegments, dtype=bool)

        for i, segment in enumerate(segments):

            # Find indices where elements should be inserted to maintain order.
            # numpy.searchsorted(a, v, side='left', sorter=None)
            # x_vals[i-1] < segment.iloc[0] <= x_vals[i]

            # fmin_index, fmax_index: Get the grid index where the x data's maximum and minumim values fall onto
            # nf: Number of grids that the recession segment covered
            fmin_index, fmax_index, nf = get_indices(Q_vals, segment, num_Q)

            # If it is too short, flag it
            if nf <= 0:
                short_segs[i] = True

        # Check if any segments are left after flagging
        if np.all(short_segs):
            # If all segements are short, return no data as MRC
            mrc_array = [np.nan, np.nan]
            fig = None
            return mrc_array, fig

        else:
            # Drop short segments and re-define the values
            segments = [seg for i, seg in enumerate(segments) if not short_segs[i]]
            numsegments = len(segments)

            # Remove flow values for interpolation if reduced segments no longer cover those values
            max_Q = max([np.max(seg) for seg in segments])
            min_Q = min([np.min(seg) for seg in segments])
            Q_vals = Q_vals[(Q_vals <= max_Q) & (Q_vals >= min_Q)]
            num_Q = len(Q_vals)

        # _________________________________________________________________________________
        # Set up the optimization matrix
        msp_matrix = np.zeros((numsegments * num_Q * 2, 3))
        b_matrix = np.zeros(numsegments * num_Q)
        mcount = 0
        mspcount = 0
        bad_segs = []

        # _________________________________________________________________________________
        # Extract and interpolate each segment
        for i, _segment in enumerate(segments):

            # Truncate the nan data at the beginning (important for satellite data)
            segment = _segment.loc[_segment.first_valid_index() :]

            if segment.iloc[0] < running_min:
                # TODO: this part is not tranlated from MATLAB
                # if segment(1) < running_min
                #     segment = [running_min , segment];
                # end
                # if there is a gap between previous segments and this one,
                # then interpolate with a vertical line

                # Find indices of max and min interpolated flow values for this segment
                first_index = segment.index.min()
                running_min_series = pd.DataFrame(
                    [running_min + jitter_size],
                    index=[first_index - 1],
                    columns=[segment.name],
                )

                # Concatenate running_min as the first row and reset the index
                segment = pd.concat([running_min_series, segment])
                # segments[i] = segment  # Replace it

            # ______________________________________
            # Get the index of flow segment in x grids
            fmin_index, fmax_index, nf = get_indices(Q_vals, segment, num_Q)

            # if no interpolated values (occurs when min and max of segment are too close together
            # Collect segment number, Don't add to minimzation matrix
            if nf == 0:
                bad_segs.append(i)
                continue

            # ______________________________________
            # Make sure the data type is correct
            y_vals = segment.values  # Get the values from the DataFrame

            # If y_vals has shape (nrows, 1), squeeze to (nrows,)
            if len(y_vals.shape) > 1 and y_vals.shape[1] == 1:
                y_vals = np.squeeze(y_vals)

            # ______________________________________
            # interpolate each segment onto the flow values
            interp_segment = interp1d(
                x=y_vals,
                y=np.arange(0, len(segment)),
                kind="linear",
                fill_value="extrapolate",
            )(Q_vals[fmax_index:fmin_index])
            running_min = min(running_min, Q_vals[fmin_index - 1])

            # ______________________________________
            # Construct the minimization matrix block for each segment

            # msp_matrix defines the structure of the system to be solved, encoding the flow-segment-to-MRC relationships and their lags in a sparse matrix format.
            # The first column of msp_matrix represents the row indices.
            # The second column contains the segment index or flow index associated with that row.
            # The third column contains coefficients (either +1, −1, or 0) that specify
            # how much the difference between the flow in the segment and the flow on the MRC
            # at a given point contributes to the total optimization.

            if i == 0:
                # Lag of the first segment is set to zero
                msp_matrix[mspcount : mspcount + nf, :] = np.c_[
                    np.arange(mspcount, mspcount + nf),
                    np.arange(numsegments + fmax_index, numsegments + fmin_index),
                    -np.ones(nf),
                ]
                b_matrix[mcount : mcount + nf] = interp_segment
            else:
                # Lags of other segments can be minimised, along with the fitted MRC
                msp_matrix[mspcount : mspcount + 2 * nf, :] = np.c_[
                    np.r_[
                        np.arange(mcount, mcount + nf), np.arange(mcount, mcount + nf)
                    ],  # Column 1
                    np.r_[
                        (i - 1) * np.ones(nf),  # First element of Column 2
                        np.arange(
                            numsegments + fmax_index, numsegments + fmin_index
                        ),  # Second element of Column 2
                    ],
                    np.r_[np.ones(nf), -np.ones(nf)],  # Column 3
                ]
                b_matrix[mcount : mcount + nf] = interp_segment

            # update count of rows in the optimisation matrix
            mcount += nf
            mspcount += nf if i == 0 else 2 * nf

            if np.isnan(b_matrix).sum() != 0:
                print("stop")

        # _________________________________________________________________________________
        # Create sparse matrix
        msp_matrix = msp_matrix[:mspcount]
        m_sparse = csr_matrix(
            (
                msp_matrix[:, 2],
                (msp_matrix[:, 0].astype(int), msp_matrix[:, 1].astype(int)),
            ),
            shape=(mcount, numsegments + num_Q),
        )

        # Cut off unused rows of optimization matrix
        B_mat = -b_matrix[:mcount]

        # Delete unused columns of minimization matrix
        bad_segs = np.array(bad_segs)
        if len(bad_segs) > 0:
            # Delete columns from m_sparse corresponding to bad segments
            row_indices = np.delete(
                np.arange(m_sparse.shape[1]), np.array(bad_segs) - 1
            )
            m_sparse = m_sparse[:, row_indices]
            # Update the number of segments
            numsegments -= len(bad_segs)

        # Minimize differences to Master Recession Curve (MRC)
        # Compute least-squares solution to equation Ax = b

        # The system of equations solved can be written as:
        #    m_sparse * X = B_mat
        # Where:
        # m_sparse is the sparse matrix encoding the flow-segment relationships.
        # X is the vector of unknowns, which includes the lags l_i and the MRC flow values.
        # B_mat is the vector of interpolated segment values. (flow values in the x grid space)

        # l_i = argmin_{l_i} sum_{k=1}_{nf} (segment_i(k)-MRC(tk+l_i))^2

        mrc_solve = np.linalg.lstsq(m_sparse.toarray(), B_mat, rcond=None)[0]

        # Extract the time lags and flow values for the MRC
        lags = np.concatenate([[0], mrc_solve[:numsegments]])
        mrc_time = mrc_solve[numsegments:]
        mrc_time = np.sort(mrc_time)
        offset = np.min(mrc_time)
        mrc_time -= offset
        lags -= offset

        # Output
        mrc_array = np.c_[mrc_time, Q_vals]

        mrc_raw_array = concat_segment_array(segments, lags)

        # _________________________________________________________________________________
        # Optional plotting
        if plot_results:
            fig = plot_mrc_fit(segments, lags)
        else:
            fig = None

        return mrc_array, mrc_raw_array, fig

    else:
        raise ValueError(
            "Fit method for MRC not recognized. Choose 'exponential' or 'nonparametric'."
        )


def concat_segment_array(segments, lags):
    # Initialize lists to store the concatenated x-values and segment values
    Q_values = []
    segment_values = []

    # Loop through the segments and lags to generate and store the x-values and segment values
    for i, segment in enumerate(segments):
        # Generate x-values based on the segment length and corresponding lag
        try:
            x = np.arange(1, len(segment) + 1) + lags[i]
        except Exception as e:
            print(f"Error generating x-values for segment {i}: {e}")
            continue

        # Append the x-values and corresponding segment values
        Q_values.append(x)
        segment_values.append(
            segment.values
        )  # Assuming 'segment' is a pandas DataFrame or Series

    # Convert the lists to numpy arrays and concatenate them vertically
    Q_values = np.concatenate(Q_values)
    segment_values = np.concatenate(segment_values)

    # Create a final array where the first column is x-values and the second column is the segment values
    _giant_array = np.column_stack((Q_values, segment_values))

    # Filter out rows where either column has a NaN
    giant_array = _giant_array[~np.isnan(_giant_array).any(axis=1)]

    return giant_array


def plot_mrc_fit(segments, lags, title="Nonparametric MRC fit"):
    """
    Plots the nonparametric MRC fit along with individual segments.

    Parameters:
    - segments: List of pandas Series or arrays representing each segment to plot
    - lags: List of lags corresponding to each segment
    - title: Title of the plot (optional)
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(6, 5))

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Data")

    timestep = 24

    # Plot each segment with its corresponding lag
    for i, segment in enumerate(segments):
        try:
            # Plot the segment as a grey line
            valid_mask = ~np.isnan(segment)  # Mask for valid (non-NaN) values

            x_data = (np.arange(1, len(segment) + 1) + lags[i]) / timestep
            y_data = segment

            ax.plot(x_data[valid_mask], y_data[valid_mask], "grey", alpha=0.3)

            # Plot the start point as a red dot
            ax.plot(
                (lags[i] + 1) / timestep, segment.iloc[0], "ro", markersize=3, alpha=0.3
            )  # Start point

            # Plot the end point as a red dot
            ax.plot(
                (lags[i] + len(segment)) / timestep,
                segment.iloc[-1],
                "ro",
                markersize=3,
                alpha=0.3,
            )  # End point
        except Exception as e:
            print(f"Error plotting segment {i}: {e}")
            continue

    # Optional: plot the fitted MRC (commented out, add the data if available)
    # ax.plot(mrc_time, Q_vals, "red", linewidth=2, label="Fitted MRC", alpha=0.5)

    # Optional: Add legend (uncomment if plotting the MRC fit)
    # ax.legend()

    # Tight layout
    fig.tight_layout()
    # fig.show()

    return fig
