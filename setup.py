# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ModelZoo - Computer Vision Model Deployment Library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies (Python 3.12+, OpenCV 4.12)
install_requires = [
    "numpy>=1.26.0",
    "opencv-python>=4.12.0",
    "onnxruntime>=1.18.0",
    "spacemit-ort",
    "Pillow>=10.0.0",
    "scipy>=1.11.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "filterpy>=1.4.5",
    "lap>=0.5.12",
    "cython-bbox>=0.1.5",
]

setup(
    name="modelzoo-vision",
    version="0.1.0",
    author="ModelZoo Team",
    author_email="",
    description="Computer Vision Model Deployment Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ModelZoo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.12",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.8",
            "black>=24.0",
        ],
    },
)

