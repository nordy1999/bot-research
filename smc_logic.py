"""
AlmostFinishedBot - SMC Logic (compatibility wrapper)
FIX: Bridge imports `from smc_logic import get_bias` — this file was missing.
Re-exports from smc_detector.py so all existing imports work.
"""
from smc_detector import get_bias, check_smc_signal, run_smc_scan
