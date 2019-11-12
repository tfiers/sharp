<p align="center">
  <img src="logo.png" alt="Logo for this project: a stylized rat and the name "Sharp">
  <br>
  <sup>A warm thank you to <a href="https://github.com/zuurw">@zuurw</a> for the free logo design.</sup>
</p>

Software written for my [master's thesis](https://github.com/tfiers/master-thesis).

This code takes raw neural recordings as input, and generates the figures found
in the thesis (and more) as output. In between, it processes the raw
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


### 3. Install

Next, install the `sharp` package (and its dependencies that are publicly 
available on PyPI):
```bash
~/code/sharp$  pip install -e .
```

Optionally enable tab-autocompletion for `sharp` commands by adding the
following line to your `.bashrc`:
```sh
. ~/code/sharp/sharp/cmdline/enable-autocomplete.sh
```


### 4. Test

You can verify whether the installation was succesful by trying on the command
line:

```bash
$ sharp
```

A message starting with 
```
Usage: sharp <options> <command>
```
should appear.


Another way to test is to run Python and try:

```py
import sharp
```


### Notes

- This installation procedure has been tested on Windows 10 and Ubuntu 16.04.
- If no GPU acceleration is desired, the significantly smaller CPU-only 
  version of PyTorch may be installed (i.e. `pytorch-cpu`, which corresponds 
  to `CUDA = None` on the PyTorch "get-started" page).
- The installation of `fklab-python-core` might fail while trying to build its
  `radonc` extension (especially on Windows). This extension is not used in
   sharp, and can be excluded from the install: edit `fklab-python-core/setup.py`,
   and remove the line `ext_modules = [radon_ext]` in the `setup()` call.




<br>
<br>

## Usage

### 1. Configuration

In your terminal, run `sharp config`, passing the name of a new directory in
which a run configuration, logs, and (optionally) task output files will be
stored:

```bash
$ sharp config ~/my-sharp-cfg
```

Edit the newly created "`config.py`" file in this directory and change the
settings (such as the location of raw data and output directories) to suit your
needs.

- See the [config specification](/sharp/config/spec.py) for explanations of the
    different options.
- See the [test `config.py` file](/tests/system/config.py) from this repository
    for a concrete example of a customized configuration.
- Set `central_server = None` to test without having to start a [central server](#3-central-server)
    first.


### 2. Running tasks

When your `config.py` file is ready, run "`sharp worker`", passing the name
of your config directory:
```bash
$ sharp worker ~/my-sharp-cfg
```
This will run the tasks specified in the `get_tasks` method of your `config.py`
file (these tasks typically generate figures), together with the tasks on which
they depend (typically processing raw data, training neural networks,
calculating evaluation metrics, ...).

**Notes**
- `sharp` internally outsources task dependency resolution and scheduling to [Luigi](https://luigi.readthedocs.io).
- On Linux-like operating systems, by default, as many subprocesses as CPU's
    will be launched, to run tasks in parallel. The number of subprocesses can
    be set explicitly with the `-n` or `--num-subprocesses` option.
- On Windows, running Luigi tasks in subprocesses is [not supported](https://github.com/spotify/luigi/pull/2720),
    and `sharp worker` will therefore run only one task at a time. If task
    parallelization on Windows is desired, run a central Luigi server (see
    below), and start multiple `sharp worker` processes.



### 3. Central server

Sharp can be used without a central Luigi server. Running [such a server](https://luigi.readthedocs.io/en/stable/central_scheduler.html)
however provides:
1. **Visualization** of task completion progress, and the task dependency graph;
2. A centralized task scheduler, to distribute task execution over multiple
    nodes in a **computing cluster** (or over multiple Python processes on
    Windows).

Sharp provides a utility script, "`sharp scheduler start`", to configure and
start a central Luigi server as a background (daemon) process. It takes as
argument the name of a new directory in which the server logs, task history
database, and scheduler PID and state files will be stored:
```bash
$ sharp scheduler start ~/luigi-scheduler
```

Similarly, "`sharp scheduler stop`" and "`sharp scheduler state`" commands are
provided.

> The Luigi scheduling server can only run as a daemon on Linux-like operating
systems. If you want to use the scheduling server on Windows, run it manually,
using the [`luigid`](https://luigi.readthedocs.io/en/stable/central_scheduler.html)
command without the `--background` option.

On each host of your computing cluster, simply start a
```bash
$ sharp worker ~/my-sharp-cfg
```
process.

Sharp provides a utility script, "`sharp slurm`", to automatically start
workers on different nodes, if [SLURM](https://slurm.schedmd.com/overview.html)
is used to manage jobs on the computing cluster:
```bash
$ sharp slurm ~/my-sharp-cfg --nodes=2
```
This:
1. submits a well-behaved Slurm job that runs "`sharp worker ~/my-sharp-cfg`" on
    two cluster nodes;
2. shows the Slurm job queue and job details.
