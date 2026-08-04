#!/usr/bin/env python3
"""
Setup script for DNA Transport GNN (G3NAT).
"""

from setuptools import setup, find_packages


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


setup(
    name="g3nat",
    version="0.2.0",
    author="William Livernois",
    author_email="willll@uw.edu",
    description="GNNs for DNA transport with PyTorch Geometric",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wliverno/G3NAT",
    packages=find_packages(include=["g3nat", "g3nat.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 