import codecs
import re
from pathlib import Path

from setuptools import setup, find_packages

PROJECT_ROOT = Path(__file__).parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
README_FILE = PROJECT_ROOT / "README.md"
VERSION_FILE = PROJECT_ROOT / "mombf" / "__init__.py"


def get_requirements(path):
    with codecs.open(path) as buff:
        return buff.read().splitlines()

def get_long_description():
    with codecs.open(README_FILE, "rt") as buff:
        return buff.read()

def get_version():
    lines = open(VERSION_FILE, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (VERSION_FILE,))

setup(
    name="mombf",
    license="GPLv3",
    version=get_version(),
    description="JAX powered version of R package mombf",
    author="Oriol Abril-Pla",
    url="http://github.com/oriolabril/mombf.py",
    packages=find_packages(),
    install_requires=get_requirements(REQUIREMENTS_FILE),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)
