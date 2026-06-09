"""
Shim — Streamlit Cloud still points to this filename.
The actual app lives in kemercayir_daily_model.py.
"""
import importlib.util, os, sys

_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "kemercayir_daily_model",
    os.path.join(_dir, "kemercayir_daily_model.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
