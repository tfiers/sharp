<p align="center">
  <img src="logo.png" alt="Logo for this project: a stylized rat and the name "Sharp">
  <br>
  <sub>Logo graciously furnished by <a href="https://github.com/zuurw">@zuurw</a>.</sub>
</p>

Software written for my [master's thesis](https://github.com/tfiers/master-thesis) 
and for the related [journal paper](https://github.com/tfiers/neural-network-paper).

The name of this Python package, `sharp`, comes from the _sharp wave-ripple_,
the electrical brain motif related to memories and learning that is studied 
in the thesis.


## Documentation

README's are provided for most sub-packages,
and docstrings are provided for most modules, classes, and methods.

Care is taken to organize the code, and to name objects in a logical way.

Also see the _Usage_ section below.


## Installation

*(Also see the installation __Notes__ below).*

The software is written in Python 3.7, and requires recent installations of
[SciPy](https://scipy.org/) and [PyTorch](https://pytorch.org/).
These are most easily installed with the [(mini)conda package manager](https://conda.io/docs/index.html).

When these dependencies are installed, clone this repository to your computer
(e.g. to a directory `~/code/sharp` as in this example):
```sh
$ git clone git@github.com:tfiers/sharp.git ~/code/sharp
```

Then, install custom-made dependencies from their respective git repositories:
```sh
~/code/sharp$  pip install -r requirements.txt
```

> For now the dependency `fklab-python-core` is closed source, and needs to be
installed manually. Request access to its git repository by contacting
[Kloosterman Lab](https://kloostermanlab.org/). Clone the repository and enter
its directory, and install with `pip install .`. Verify that it is installed
correctly by trying `import fklab` in Python. See also the notes below.

Next, install this package (and its dependencies from [PyPI](https://pypi.org/)):
```sh
~/code/sharp$  pip install -e .
```

You can verify whether the installation was succesful by running Python and
trying:
```py
import sharp
```

#### Notes

- This installation procedure has been tested on Windows 10 and Ubuntu 16.04.
- If no GPU acceleration is desired, the significantly smaller CPU-only 
  version of PyTorch may be installed (i.e. `pytorch-cpu`, which corresponds 
  to `CUDA = None` on the PyTorch "get-started" page).
- The installation of `fklab-python-core` might fail while trying to build its
  `radonc` extension (especially on Windows). This extension is not used in my
  thesis, and can be excluded from the install: edit `fklab-python-core/setup.py`, 
  and remove the line `ext_modules = [radon_ext]` in the `setup()` call.



## Usage


Create a new directory to store run configuration, logs, and (optionally) output
files:
```sh
$ mkdir ~/sharp-run
```

Optionally store this directory path to an environment variable named
`SHARP_CONFIG_DIR`:
```sh
$ export SHARP_CONFIG_DIR=~/sharp-run
```
(This is not necessary if you will always run `sharp` from within this new
directory.)

In the new directory, create a file named `config.py`, containing a class named
`SharpConfig` that subclasses `SharpConfigBase` from [`sharp.config.spec`](sharp/config/spec.py).
Change some or all of the parent attributes to suit your needs.

See the test [`config.py`](tests/system/config.py) file from this repository
for an example.

> On Windows, make sure to either use forward slashes in paths, or to escape
backslashes. Examples:
```toml
raw_data_dir = "D:/data/probe/L2"
output_dir = "subdir/of/current/working/directory"
logging_conf_file = "D:\\code\\sharp\\logging.cfg"
```

When the `config.py` file is ready, process the raw data and generate figures
and other output by running:
```sh
~/sharp-run$  python -m sharp --local-scheduler
```

Show command documentation with:
```sh
$ python -m sharp --help
```

#### Parallelization

To run subtasks in parallel, a central Luigi task scheduler should be used
instead of the local scheduler. See [here](https://luigi.readthedocs.io/en/stable/central_scheduler.html)
for instructions.

When the central scheduler is running, and when you have correctly set the
`luigi_scheduler_host` setting in your `config.py` file, simply start multiple
`python -m sharp` processes.
