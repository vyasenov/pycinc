"""
PyCIC: Changes-in-Changes Model Implementation

A Python package for estimating the Changes-in-Changes (CiC) model 
introduced by Athey and Imbens (2006).
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .cic import ChangesInChanges
from .results import CiCResults

__all__ = ["ChangesInChanges", "CiCResults"] 