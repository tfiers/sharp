from setuptools import find_packages, setup

description = """
Algorithms and evaluation code for real-time sharp wave-ripple detection.
"""

setup(
    name="sharp",
    version="1.0",
    description=description,
    author="Tomas Fiers",
    author_email="tomas.fiers@gmail.com",
    license="GPL-3.0",
    install_requires=(
        "torch",
        "toml",
        "h5py",
        "click",
        "typeguard",
        # Version pin of python-daemon is necessary to make Luigi install work
        # on Windows:
        "python-daemon==2.1.2",
        # Sqlalchemy is a Luigi dependency (but weirdly not in their setup.py).
        "sqlalchemy",
        "luigi[toml]==2.8.5",
        "scikit-learn",
        # All from requirements.txt:
        "seaborn==0.9.0+tomas",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
    entry_points={"console_scripts": ["sharp=sharp.cli.main:main_cli"]},
)
