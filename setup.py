from setuptools import setup, find_packages

name: str = "nistrng"
version: str = "1.0.0"
requirements: [] = ["numpy>=1.14.5", "scipy>=1.2.1"]
packages: [] = find_packages()
url: str = "https://github.com/InsaneMonster/NistRng"
lic: str = "Creative Commons CC-BY 3.0"
author: str = "Luca Pasqualini"
author_email: str = "psqluca@gmail.com"
description: str = "University of Siena Random Number Generator NIST Test Suite - SAILab"

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
)
