# -*- coding: utf-8 -*-
from setuptools import setup

setup(name="water",
      version=0.01,
      maintainer="Jörg Buchwald",
      maintainer_email="joerg_buchwald@ufz.de",
      author="Jörg Buchwald",
      author_email="joerg.buchwald@ufz.de",
      url="https://github.com/joergbuchwald/mini-projects",
      classifiers=["Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Visualization",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Scientific/Engineering :: Mathematics",
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8"],
      license="BSD-3 -  see LICENSE.txt",
      platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
      include_package_data=True,
      python_requires='>=3.8',
      install_requires=["numpy"],
      py_modules=["water/water"],
      packages=["water/properties"])
