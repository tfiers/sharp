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
        "python-daemon==2.1.2",  # For luigi install on Windows.
        "luigi[toml]",
        "scikit-learn",
        "raincloud",
        "fklab-python-core",
    ),
    packages=find_packages(),
)
