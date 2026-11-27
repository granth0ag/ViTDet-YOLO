"""Optional run names keep independent experiments out of legacy checkpoint paths."""
import argparse
import re


def run_name(value):
    if value == "":
        return ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError("run name must contain only letters, digits, underscores or hyphens")
    return value


def run_suffix(name):
    return "_run-" + run_name(name) if name else ""


def positive_float(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number
