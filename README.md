# _sharp_

Software written for my [master's thesis](https://github.com/tfiers/master-thesis).

The name of this Python package, `sharp`, comes from the _sharp wave-ripple_,
the electrical brain motif related to memories and learning that is studied 
in the thesis.


## Documentation

Docstrings are provided for most modules, classes, and methods.

Care is taken to name objects and organize the code in a logical way.


## Installation

The software is written in Python 3.7, and requires recent installations of
[SciPy](https://scipy.org/) and [PyTorch](https://pytorch.org/).
These are most easily installed with the [(mini)conda package manager](https://conda.io/docs/index.html).

Clone this git repository to your computer:
```sh
cd ~/code
git clone 
```

Then:
```sh
pip install -r requirements.txt
```
(This installs custom-made dependencies from their respective git repositories).

> For now the dependency `fklab-python-core` is closed source, and needs to be
installed manually. Request access to its git repository by contacting
[Kloosterman Lab](https://kloostermanlab.org/). Clone the repository and enter
its directory, and install with `pip install .`. Verify that it is installed
correctly by trying `import fklab` in Python. See also the notes below.

Next, install this package (and additional PyPI dependencies) by running:
```sh
pip install -e .
```

You can verify whether the installation was succesful by running Python and
trying:
```py
import sharp
```

To actually apply the software to data, you need to tell it where this data can
be found, and where it may store output files. See the _Usage_ section below.


#### Notes

- This installation procedure has been tested on Windows 10 and Ubuntu 16.04.
- If no GPU acceleration is desired, the significantly smaller CPU-only 
  version of PyTorch may be installed (i.e. `pytorch-cpu`, which corresponds 
  to `CUDA = None` on the PyTorch "get-started" page).
- On succesfully installing `fklab-python-core`:
     - The install might fail when trying to build the `radonc` extension.
       This extension is not used in my thesis, and can be excluded from
       the install: edit `fklab-python-core/setup.py`, and remove the line
       `ext_modules = [radon_ext]` in the `setup()` call.
     - Additionally, the dependency `spectrum` might fail to build
       (especially on Windows). Again, this dependency is not necessary,
       and can be removed from `fklab-python-core` as follows:
       in the `setup()` call in `setup.py`, remove `spectrum` from the
       `install_requires` list; and in `fklab/plot/core/artists.py`,
       remove the `import fklab.signals.multitaper` line.



## Usage

Create a new directory to store run configuration, logs, and (by default) output
files:
```sh
mkdir ~/sharp-run
```

Set the following environment variable:
```sh
export LUIGI_CONFIG_PARSER=toml
```

Create a run configuration file named `luigi.toml`, using 
[TOML syntax](https://github.com/toml-lang/toml#readme).
The [test config file](https://github.com/tfiers/sharp/blob/master/tests/system/luigi.toml)
from this repository's system test directory can be used as a starting point:
```sh
cp ~/code/sharp/tests/system/luigi.toml ~/sharp-run
vim ~/sharp-run/luigi.toml
```

> On Windows, make sure to either use forward slashes in paths, or to escape
backslashes. Example:
```toml
raw_data_dir = "D:/data/probe/L2"
output_dir = "subdir/of/current/working/directory"
logging_conf_file = "D:\\code\\sharp\\logging.cfg"
```

When the `luigi.toml` config file is tailored to your needs, process the raw
data and generate figures by running:
```sh
cd ~/sharp-run
python -m sharp --local-scheduler
```

Show command documentation with:
```sh
python -m sharp --help
```


### Parallelization

To run subtasks in parallel, a central Luigi task scheduler should be used
instead of the local scheduler. See [here](https://luigi.readthedocs.io/en/stable/central_scheduler.html)
for instructions.

When the central scheduler is running, simply start multiple `python -m sharp` 
processes from the directory containing the `luigi.toml` configuration file.

To run multiple configurations (each with their own run directory and 
`luigi.toml` file) in parallel, run each Python process with a different setting
for the `LUIGI_TASK_NAMESPACE` environment variable. Example:
```sh
cd ~/sharp-run-two-RNN-layers
LUIGI_TASK_NAMESPACE=two-layers python -m sharp
```
