"""analytical solution for thermo-osmosis problem in 1D"""

from setuptools import setup, find_packages

setup(name="zhou_solution_thermo_osmosis.py",
      version=0.1,
      maintainer="Jörg Buchwald",
      maintainer_email="joerg_buchwald@ufz.de",
      author="Jörg Buchwald",
      author_email="joerg.buchwald@ufz.de",
      url="https://github.com/joergbuchwald/thermo-osmosis_analytical_solution",
      platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
      include_package_data=True,
      install_requires=["numpy", "scipy"],
      py_modules=["zhou_solution_thermo_osmosis"],
      packages=[])
