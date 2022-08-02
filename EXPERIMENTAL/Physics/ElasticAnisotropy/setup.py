"""Elastic Anisotropy"""

from setuptools import setup, find_packages



setup(
    name="ElasticAnisotropy",
    version=0.1,
    maintainer="Jörg Buchwald",
    maintainer_email="joerg_buchwald@ufz.de",
    author="Jörg Buchwald",
    author_email="joerg.buchwald@ufz.de",
    url="https://github.com/joergbuchwald/",
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    include_package_data=True,
    install_requires=["numpy"],
    py_modules=["ElasticAnisotropy"])
