"""OSCC PNI research pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oscc-pni")
except PackageNotFoundError:  # Source checkout without installation.
    __version__ = "0.1.0"

__all__ = ["__version__"]
