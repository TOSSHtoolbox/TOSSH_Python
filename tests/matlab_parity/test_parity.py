"""Compare Python TOSSH outputs against reference values computed with the MATLAB version.

The cases are defined in cases.json. The reference values are stored in reference/<case name>.mat and are created
with generate_reference.py (requires MATLAB). Running these tests does not require MATLAB.
"""
import copy
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

here = pathlib.Path(__file__).resolve().parent
repo = here.parents[1]
sys.path.insert(0, str(repo))  # makes TOSSH_code importable regardless of where pytest is started
data_file = repo / "example" / "example_data" / "33029_daily.csv"
reference_dir = here / "reference"
cases = json.loads((here / "cases.json").read_text())

# relative tolerance for comparing Python and MATLAB outputs
rtol = 1e-6


def load_data():
    data = pd.read_csv(data_file)
    series = {column: data[column].values for column in data.columns}
    series["t"] = pd.to_datetime(data["t"], format="%d-%b-%Y").values
    return series


def load_reference(name):
    # returns the MATLAB outputs of a case as a list of arrays, or None if there is no reference file
    reference_file = reference_dir / f"{name}.mat"
    if not reference_file.exists():
        return None
    return list(loadmat(reference_file)["outputs"].ravel())


def find_function(name):
    # look for the function in the TOSSH_code subfolders, e.g. TOSSH_code/signature_functions/sig_BFI.py
    for folder in ["signature_functions", "utility_functions", "calculation_functions"]:
        if (repo / "TOSSH_code" / folder / f"{name}.py").exists():
            module = __import__(f"TOSSH_code.{folder}.{name}", fromlist=[name])
            return getattr(module, name)
    pytest.skip(f"{name} has not been translated to Python yet")


data = load_data()


@pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])
def test_matches_matlab(case):
    reference = load_reference(case["name"])
    if reference is None:
        pytest.skip("no MATLAB reference values for this case, run generate_reference.py")

    function = find_function(case["function"])
    inputs = [data[name].copy() if isinstance(name, str) else name for name in case["inputs"]]
    outputs = function(*inputs, **copy.deepcopy(case["options"]))
    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    for i, expected in enumerate(reference):
        # strings (error_str) and empty outputs (fig_handles) are not compared
        if expected.dtype.kind not in "biuf" or expected.size == 0:
            continue
        # squeeze, as MATLAB stores all values as matrices (scalars as 1x1, vectors as nx1 or 1xn)
        expected = np.squeeze(expected.astype(float))
        actual = np.squeeze(np.asarray(outputs[i], dtype=float))
        np.testing.assert_allclose(actual, expected, rtol=rtol, err_msg=f"output {i + 1} of {case['function']}")


if __name__ == "__main__":
    # allows running this file directly (python test_parity.py) instead of via pytest
    pytest.main([__file__, "-v"])
