"""Regenerate reference/reference.json by running the MATLAB version of TOSSH.

Requires MATLAB. By default, the MATLAB TOSSH repository is expected next to this repository
(e.g. C:/Projects/_TOSSH/TOSSH). Set the environment variable TOSSH_MATLAB_PATH to use another location,
and MATLAB_EXE if matlab is not on the PATH.

Usage: python tests/matlab_parity/generate_reference.py
"""
import os
import pathlib
import subprocess

here = pathlib.Path(__file__).resolve().parent
tossh_matlab = pathlib.Path(os.environ.get("TOSSH_MATLAB_PATH", here.parents[2] / "TOSSH"))
matlab_exe = os.environ.get("MATLAB_EXE", "matlab")

if not (tossh_matlab / "TOSSH_code").is_dir():
    raise FileNotFoundError(f"MATLAB TOSSH not found at {tossh_matlab}. Set TOSSH_MATLAB_PATH.")

matlab_command = f"cd('{here.as_posix()}'); generate_reference('{tossh_matlab.as_posix()}')"
subprocess.run([matlab_exe, "-batch", matlab_command], check=True)
