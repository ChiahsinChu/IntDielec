"""
Setup script for `IntDielec`
"""

from setuptools import setup, find_packages

setup(
    name = "intdielec",
    version = "0.1",
    # include all packages in src
    packages = find_packages(),  
    # exclude by
    #find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
    # all packages required for this package is in src
    include_package_data = True
)

