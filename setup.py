"""
Setup script for `IntDielec`
"""

from setuptools import setup, find_packages

setup(name="intdielec",
      version="1.0",
      packages=find_packages(),
      include_package_data=True,
      install_requires=["MDAnalysis == 2.3.0", "numpy == 1.20"])
