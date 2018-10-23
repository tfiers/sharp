from setuptools import find_packages, setup

description = """
Algorithms and evaluation code for real-time sharp wave-ripple detection.
"""

setup(
    name="sharp",
    version="0.1",
    description=description,
    author="Tomas Fiers",
    author_email="tomas.fiers@gmail.com",
    license="GPL-3.0",
    install_requires=(
        "torch",
        "toml",
        "click",
        # Version pin necessary to make luigi install work on Windows:
        "python-daemon==2.1.2",
        "luigi[toml]",
        "scikit-learn",
        # All from requirements.txt:
        "seaborn==0.9.0+tomas",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
)
