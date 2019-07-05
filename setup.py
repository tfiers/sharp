from setuptools import find_packages, setup

description = """
Algorithms and evaluation code for real-time sharp wave-ripple detection.
"""

setup(
    name="sharp",
    version="2.0",
    description=description,
    author="Tomas Fiers",
    author_email="tomas.fiers@gmail.com",
    license="GPL-3.0",
    install_requires=(
        #
        # Most easily installed using conda (see "environment.yml")
        "torch >=1",
        "numpy >=1.11",
        "scipy >=0.17",
        "scikit-learn >=0.21.2",
        "matplotlib >=3.1",
        #
        # Plain PyPI packages:
        "h5py >=2.9",
        "click >=7",
        "preludio >=1",
        "cached-property >=1.5.1",
        #
        # Custom-made packages, not available on PyPI, that should be
        # installed manually before installing "sharp" (see README):
        "farao",
        "fklab-python-core >=1.1.1",
    ),
    packages=find_packages(),
    entry_points={"console_scripts": ["sharp=sharp.workflow:main"]},
)
