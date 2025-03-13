### TOSSH workflow 0 - basic workflow for Python
#
# This script shows the basic functionalities of TOSSH with some example data.
#
# The example data used in this workflow are taken from CAMELS-GB
# (Coxon et al., 2020), see README_example_data.txt for more information
# on data sources.
#
# Copyright (C) 2025
# This software is distributed under the GNU Public License Version 3.
# See <https://www.gnu.org/licenses/gpl-3.0.en.html> for details.

# load required packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load signature functions
from TOSSH_code.signature_functions.sig_BFI import sig_BFI

mydir = os.getcwd()  # Get current working directory, e.g. for loading data in subfolders
sys.path.append(mydir)  # Add current working directory to path (optional)

# Typically, users will have their own data which they want to analyse. We provide an example file to get a more
# realistic time series. The example file contains precipitation (P), potential evapotranspiration (PET), and
# temperature (T) data, which are required for some signatures.
path = mydir + '/example_data/'  # specify path

# We now load data, which is stored in a csv file.
data = pd.read_csv(path + '33029_daily.csv', delimiter=',')
t = pd.to_datetime(data['t']).values
Q = data['Q'].values  # streamflow [mm/day]
P = data['P'].values  # precipitation [mm/day]
# Note: PET and T example data are provided but not used here.
# PET = data['PET'].values  # potential evapotranspiration [mm/day]
# T = data['T'].values  # temperature [degC]
# Note that the data format is assumed to be an array (for t a datetime64 array and for the other data a float array).
# Other formats may also be used (e.g. lists, pandas Series/DataFrames), but would require adjustments in the code.

# Plot data
# We can plot the data to get a first idea of the hydrograph.
plt.figure(figsize=(7, 4))
plt.plot(t, Q, 'k-', linewidth=1.0)
plt.xlabel('Date')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
plt.ylabel('Streamflow [mm/day]')
plt.show()

# Calculate signatures
# Once the input data are loaded, we can calculate different signatures. We start by calculating the baseflow index
# (BFI). The default method is the UKIH smoothed minima method with a parameter of 5 days.
BFI_UKIH, _, _, _ = sig_BFI(Q, t)
print(f'BFI_UKIH = {BFI_UKIH:.2f}')
# Alternatively, we can use the Lyne-Hollick filter with a filter parameter of 0.925.
BFI_LH, _, _, _ = sig_BFI(Q, t, method='Lyne_Hollick')
print(f'BFI_LH = {BFI_LH:.2f}')
# We can also change the parameter value of the UKIH method to 10 days. Note that parameters needs to be added as list.
BFI_UKIH10, _, _, _ = sig_BFI(Q, t, method='UKIH', parameters=[10])
print(f'BFI_UKIH10 = {BFI_UKIH10:.2f}')
# As we can see, all three options lead to slightly different values. More details and examples on the different
# methods/parameters can be found in the code of each function (e.g. sig_BFI.m).

# Warnings and errors
# Each signature function can return a warning/error output. These warning/error outputs indicate problems during
# signature calculation, but they do not stop code execution like a normal Matlab error would do. Two outputs can be
# retrieved: an error flag (error_flag) that corresponds to a certain type of warning/error, and a string (error_str)
# that decribes the warning/error. If multiple warnings/errors occur, they are all listed in the error string, starting
# with the one that occurred last.

# A warning (error_flag = 1) typically indicates that the signature can be calculated but should be interpreted with
# care, e.g. because there are NaN values in the time series. To retrieve the error flags and other outputs beyond the
# signature values, we need to specify the number of output arguments (nargout) in the run_tossh_function. For example,
# in the following it is set to 3, so that we can retrieve the error flag and error string. Functions that return even
# more outputs can be handled in the same way by setting nargout accordingly.
Q[:10] = np.nan
BFI, error_flag, error_str, _ = sig_BFI(Q, t)
print(f'BFI (with NaN values) = {BFI:.2f}')
print(error_str)
# We get the same mean value as before since the ten removed values do not influence the result much. In other cases,
# significant amounts of NaN entries might cause more problems.

# An error (error_flag > 1) indicates that the signature could not be calculated, e.g. because there is a problem with
# the input data. For example, if the input time series contains negative and thus physically impossible values, NaN is
# returned.
Q[:10] = -1.0
BFI, error_flag, error_str, _ = sig_BFI(Q, t)
print(f'BFI (with negative Q values) = {BFI:.2f}')
print(error_str)

# Since these warnings/errors do not stop the execution of the code, we can run the signature code for many catchments
# without breaking, even if for some of the catchments the signature cannot be calculated. There may still be "normal"
# erros which can happen if the input parameters are specified incorrectly (wrong format, wrong range, etc.). These
# errors will stop the code execution as usual.

# Further information
# Further information can be found in the online documentation: https://TOSSHtoolbox.github.io/TOSSH/
