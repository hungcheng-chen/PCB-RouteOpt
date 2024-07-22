#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    author="HungCheng Chen",
    author_email="hcchen.nick@gmail.com",
    name="pcb_routeopt",
    keywords="pcb_routeopt",
    packages=find_packages(
        include=["pcb_routeopt", "pcb_routeopt.*"]
    ),
    test_suite="tests",
    license="MIT license",
    version="0.1.0",
)
