# _sharp_

Software written for my [master's thesis](https://github.com/tfiers/master-thesis).

The name of this Python package, `sharp`, comes from the _sharp wave-ripple_,
the electrical brain motif related to memories and learning that is studied 
in the thesis.


## Documentation

Docstrings are provided for most modules, classes, and methods.

Care is taken to organize the code and name objects in a logical way.

Also see the _Usage_ section below.


## Installation

The software is written in Python 3.7, and requires recent installations of
[SciPy](https://scipy.org/) and [PyTorch](https://pytorch.org/).
These are most easily installed with the [(mini)conda package manager](https://conda.io/docs/index.html).

Clone this repository to your computer:
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

Create a new directory to store run configuration, logs, and (optionally) output
files:
```sh
$ mkdir ~/sharp-run
```

In this directory, create a [Luigi run configuration file](https://luigi.readthedocs.io/en/stable/configuration.html),
named `luigi.toml`.

The [test `luigi.toml` file](https://github.com/tfiers/sharp/blob/master/tests/system/luigi.toml)
from this repository can be used as a template:
```sh
$ cp ~/code/sharp/tests/system/luigi.toml ~/sharp-run
$ vim ~/sharp-run/luigi.toml
```

> On Windows, make sure to either use forward slashes in paths, or to escape
backslashes. Examples:
```toml
raw_data_dir = "D:/data/probe/L2"
output_dir = "subdir/of/current/working/directory"
logging_conf_file = "D:\\code\\sharp\\logging.cfg"
```

Set the following environment variable:
```sh
$ export LUIGI_CONFIG_PARSER=toml
```

When the `luigi.toml` config file is tailored to your needs, process the raw
data and generate figures by running:
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

When the central scheduler is running, simply start multiple `python -m sharp` 
processes from the directory containing the `luigi.toml` configuration file.

You can also run multiple configurations in parallel (each with their own run
directory and `luigi.toml` file). To do this, make sure the
`SharpConfig.config_id` setting has a unique value in each `luigi.toml` file.
Then simply start Python processes from the respective run directories.
