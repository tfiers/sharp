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
        "torch",
        "h5py",
        "scikit-learn",
        #
        # Custom-made dependencies, not available on PyPI, that should be
        # installed manually before installing sharp:
        #   pip install -r requirements.txt
        "seaborn==0.9.0+tomas",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
)
