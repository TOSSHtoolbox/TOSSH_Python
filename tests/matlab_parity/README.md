# MATLAB parity tests

These tests check that the Python translation gives the same results as the original
[MATLAB TOSSH](https://github.com/TOSSHtoolbox/TOSSH). The MATLAB outputs are computed once and stored in
`reference/` (one `.mat` file per case), so running the tests does not require MATLAB.

## Files

| File | Purpose |
| --- | --- |
| `cases.json` | Test cases: function name, input time series and options. Used by both MATLAB and Python. |
| `generate_reference.py` | Runs `generate_reference.m` in MATLAB to (re)create the reference values. |
| `generate_reference.m` | Calls the MATLAB functions for each case and writes one `reference/<case name>.mat` per case. |
| `reference/<case name>.mat` | MATLAB outputs of a case (cell array `outputs`, loaded in Python with `scipy.io.loadmat`). |
| `reference/info.json` | MATLAB version and TOSSH commit the reference values were created with. |
| `test_parity.py` | Runs each case in Python and compares the outputs with the reference values. |

All cases use the example data in `example/example_data/33029_daily.csv`.

## Running the tests

```
`python -m pytest -v`            runs tests for all signatures in the cases file
`python -m pytest -v -k BFI`     runs only the cases whose name contains "BFI" (or whatevery is put in the last argument position) 
```

or run `test_parity.py` directly. Each case is reported as `PASSED`, `FAILED`
(with the differing values) or `SKIPPED` (no reference values yet, or function not yet translated to Python).

Numeric outputs are compared with a relative tolerance of `1e-6`. Error strings and figure handles are not compared.

## Adding a test case

1. Add an entry to `cases.json`, e.g.

   ```json
   {"name": "BFI_UKIH_10", "function": "sig_BFI", "inputs": ["Q", "t"], "options": {"method": "UKIH", "parameters": [10]}}
   ```

   - `name`: unique name (letters, numbers and underscores only, as it is used as a MATLAB field name)
   - `function`: function name, identical in MATLAB and Python
   - `inputs`: columns of the example data passed as positional inputs (`t`, `Q`, `P`, `PET`, `T`)
   - `options`: optional name-value arguments (MATLAB) / keyword arguments (Python)

2. Regenerate the reference values (requires MATLAB):

   ```
   python tests/matlab_parity/generate_reference.py
   ```

   By default, the MATLAB TOSSH repository is expected next to this repository (e.g. `_TOSSH/TOSSH` and
   `_TOSSH/TOSSH_Python`). Otherwise, set the environment variable `TOSSH_MATLAB_PATH`. If `matlab` is not on the
   PATH, set `MATLAB_EXE` to the MATLAB executable.

3. Run the tests and commit `cases.json` together with the new/updated files in `reference/`.
   If you rename or remove a case, delete its old `.mat` file.

Regenerate the reference values also after updating the MATLAB repository, to check whether the Python version is
still consistent with it.
