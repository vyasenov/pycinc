"""
PyCIC: Changes-in-Changes Model Implementation

A Python package for estimating the Changes-in-Changes (CiC) model 
introduced by Athey and Imbens (2006).
"""

__version__ = "0.1.0"

from .cic import ChangesInChanges
from .results import CiCResults

__all__ = ["ChangesInChanges", "CiCResults"] 