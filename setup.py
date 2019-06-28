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
        # Ideally installed using conda (see README):
        "torch",
        "scikit-learn",  # and its dependencies: NumPy, SciPy
        #
        # Normal PyPI packages:
        "h5py",
        "click",
        #
        # Custom-made packages, not available on PyPI, that should be
        # installed manually before installing sharp:
        #   pip install -r requirements.txt
        "seaborn==0.9.0+tomas",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
    entry_points={"console_scripts": ["sharp=sharp.workflow:run_sequentailly"]},
)
