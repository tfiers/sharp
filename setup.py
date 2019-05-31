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
        "scikit-learn",
        "psutil",
        # Sqlalchemy is a Luigi dependency (but weirdly not in their setup.py).
        "sqlalchemy",
        "luigi[toml]==2.8.5",
        #
        # Custom-made dependencies, not available on PyPI, that should be
        # installed manually before installing sharp:
        #   pip install -r requirements.txt
        "seaborn==0.9.0+tomas",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
    entry_points={"console_scripts": ["sharp=sharp.cmdline.main:main_cli"]},
)
