"""ogs6py: a python API for OpenGeoSys6"""

from setuptools import setup, find_packages

setup(name="heatsource.py",
      version=0.1,
      maintainer="Jörg Buchwald",
      maintainer_email="joerg_buchwald@ufz.de",
      author="Jörg Buchwald",
      author_email="joerg.buchwald@ufz.de",
      url="https://github.com/joergbuchwald/heatsource_Thm",
      platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
      include_package_data=True,
      install_requires=["numpy", "scipy"],
      py_modules=["heatsource"],
      packages=[])
