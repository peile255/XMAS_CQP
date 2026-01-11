from setuptools import setup, find_packages
from pathlib import Path

# ---------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------

PROJECT_NAME = "xmas-cqp"
VERSION = "0.1.0"
DESCRIPTION = (
    "XMAS-CQP: Explainable Multi-Agent System for Code Quality Prediction"
)
AUTHOR = "Lei Pei"
AUTHOR_EMAIL = "lei.pei@postgrad.otago.ac.nz"
LICENSE = "MIT"
URL = "https://github.com/drleipei/xmas-cqp"

# ---------------------------------------------------------------------
# Long description (README)
# ---------------------------------------------------------------------

this_dir = Path(__file__).parent
readme_path = this_dir / "README.md"

long_description = (
    readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else DESCRIPTION
)

# ---------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------

requirements_path = this_dir / "requirements.txt"
if requirements_path.exists():
    install_requires = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
else:
    install_requires = []

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,

    packages=find_packages(exclude=("tests", "notebooks")),
    include_package_data=True,

    install_requires=install_requires,

    python_requires=">=3.10",

    entry_points={
        "console_scripts": [
            "xmas-cqp=xmas_cqp.cli:main",
        ]
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Artificial Intelligence :: Explainable AI",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],

    project_urls={
        "Source": URL,
        "Issues": f"{URL}/issues",
    },

    keywords=[
        "explainable ai",
        "code quality",
        "software engineering",
        "llm",
        "multi-agent systems",
        "xai",
    ],
)
