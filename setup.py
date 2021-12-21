import pathlib
from setuptools import setup, find_packages


base_packages = ["scikit-learn>=1.0.0", "cleanlab>=1.0", "pandas>=1.3.3"]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
    "mktestdocs==0.1.2",
]

test_packages = [
    "interrogate>=1.5.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pre-commit>=2.2.0",
    "flake8-print>=4.0.0",
    "whatlies==0.6.4",
]

all_packages = base_packages
dev_packages = all_packages + docs_packages + test_packages


setup(
    name="doubtlab",
    version="0.1.5",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Don't Blindly Trust Your Labels",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://koaning.github.io/doubtlab/",
    project_urls={
        "Documentation": "https://koaning.github.io/doubtlab/",
        "Source Code": "https://github.com/koaning/doubtlab/",
        "Issue Tracker": "https://github.com/koaning/doubtlab/issues",
    },
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
    license_files=("LICENSE",),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
