Classes that describe how the data processed and produced by this package is
stored on disk and represented in memory.

The actual data is stored somewhere else on the file system (see `config.py`).

Calculations performed in this module should be minimal (e.g property accesses
should return near-instantaneously). The time-consuming work is done in
`tasks/`.
