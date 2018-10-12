"""
A collection of `Luigi` tasks to describe entire raw data-to-figure pipelines.

These are batch jobs that take files as inputs and write files as outputs.
Luigi resolves the dependencies between these tasks, and runs them, skipping
tasks that have already been completed.

See also https://luigi.readthedocs.io
"""
