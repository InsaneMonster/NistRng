#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import setuptools

from setuptools import setup, find_packages

# Define function to read from README.rst file


def readme():
    with open("README.rst") as file:
        return file.read()


# Set all package values

name: str = "nistrng"
version: str = "1.2.3"
requirements: [] = ["numpy>=1.14.5", "scipy>=1.2.2"]
packages: [] = find_packages()
url: str = "https://github.com/InsaneMonster/NistRng"
lic: str = "BSD 3-Clause"
author: str = "Luca Pasqualini"
author_email: str = "pasqualini@diism.unisi.it"
description: str = "NIST Test Suite for Random Number Generators - SAILab - University of Siena"
long_description: str = readme()
keywords: str = "NIST Tests RNG Random Number Generator SAILab USiena Siena SP800-22r1a"
include_package_data: bool = True
classifiers: [] = [
                    "Development Status :: 5 - Production/Stable",
                    "License :: OSI Approved :: BSD License",
                    "Programming Language :: Python :: 3.6",
                    "Topic :: Security :: Cryptography",
                  ]

# Run setup

setup(
    name=name,
    version=version,
    install_requires=requirements,
    packages=packages,
    url=url,
    license=lic,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    keywords=keywords,
    include_package_data=include_package_data,
    classifiers=classifiers
)
