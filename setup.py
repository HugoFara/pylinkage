#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:04:06 2021

@author: HugoFara
"""
import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Pylinkage",
    version = "0.0.1",
    author = "Hugo Farajallah",
    author_email = "Hugo DOT Farajallah  AT protonmail.com",
    description = "A package intended to build and optimize planar linkages.",
    license = "MIT License",
    keywords = "linkage mechanism optimization",
    url = "",
    packages=['pylinkage', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering"
        ]
)