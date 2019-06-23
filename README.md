<p align="center">
  <img src="logo.png" alt='Logo for this project: a stylized rat and the name "Sharp"'>
  <br>
  <sup>A warm thank you to <a href="https://github.com/zuurw">@zuurw</a> for the free logo design.</sup>
</p>

Software written for my [master's thesis](https://github.com/tfiers/master-thesis) 
and for the related [journal paper](https://github.com/tfiers/neural-network-paper).

This code takes raw neural recordings as input, and generates the figures found
in the papers (and more) as output. In between, it processes the raw
recordings, trains neural networks, calculates detection performance metrics,
etc.

The name of this Python package, `sharp`, comes from the _sharp wave-ripple_,
the electrical brain motif related to memories and learning that is studied
in the thesis and the paper. More specifically, we seek to find new real-time
algorithms that make earlier online sharp wave-ripple detections.

Jump to: 
[Installation](#installation) |
[Usage](#usage)



<br>
<br>

## Documentation

See the [_Usage_](#Usage) section below.

On code documentation: README's are provided for most sub-packages,
and docstrings are provided for most modules, classes, and methods.
Care is taken to organize the code, and to name objects in a logical way.




<br>
<br>

## Installation

*(Also see the installation __Notes__ below).*

### 1. Download

Clone this repository to your computer (e.g. to a directory `~/code/sharp` as
in this example):
```bash
$ git clone git@github.com:tfiers/sharp.git ~/code/sharp
```

### 2. Dependencies

The software is written in Python 3.7, and requires recent installations of
[SciPy](https://scipy.org/) and [PyTorch](https://pytorch.org/).
These are most easily installed with the [conda package manager](https://conda.io/docs/index.html).
(Install either [Anaconda](https://www.anaconda.com/distribution/)
or the much smaller [miniconda](https://docs.conda.io/en/latest/miniconda.html)).

Next, install the required packages that are not publicly available on
[PyPI](https://pypi.org/):
```bash
~/code/sharp$  pip install -r requirements.txt
```
This will fetch them automatically from their respective git repositories.

> For now, the dependency `fklab-python-core` is closed source, and needs to be
downloaded and installed manually. Request access to its git repository by
contacting [Kloosterman Lab](https://kloostermanlab.org/). Clone the
repository, enter its directory, and install with `pip install .`. Verify that
it is installed correctly by trying `import fklab` in Python. See also the
notes below.


### Notes

- This installation procedure has been tested on Windows 10 and Ubuntu 16.04.
- If no GPU acceleration is desired, the significantly smaller CPU-only 
  version of PyTorch may be installed (i.e. `pytorch-cpu`, which corresponds 
  to `CUDA = None` on the PyTorch "get-started" page).
- The installation of `fklab-python-core` might fail while trying to build its
  `radonc` extension (especially on Windows). This extension is not used in
   sharp, and can be excluded from the install: edit `fklab-python-core/setup.py`,
   and remove the line `ext_modules = [radon_ext]` in the `setup()` call.
