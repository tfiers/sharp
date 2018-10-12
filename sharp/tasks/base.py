from os import environ
from typing import List

import luigi
from luigi import Target, Task
from luigi.task import flatten


class SafeTask(Task):
    """
    A luigi Task in which `output()` is only ever executed by luigi if all
    dependencies specified in `requires()` have been completed.
    """

    def complete(self) -> bool:
        return all(
            dependency.complete() for dependency in self._dependencies
        ) and all(output.exists() for output in self._outputs)

    @property
    def _dependencies(self) -> List[Task]:
        return flatten(self.requires())

    @property
    def _outputs(self) -> List[Target]:
        return flatten(self.output())


class NamespacedTask(Task):
    """
    Base class for all tasks in this package.
    When inheriting from multiple tasks, this class should be the first parent.
    """

    task_namespace = environ.get("LUIGI_TASK_NAMESPACE", "")
    # Setting this to a custom value allows to run multiple pipelines (each
    # with a different `luigi.toml` config file) in parallel. Each such
    # pipeline / config file corresponds to a different `task_namespace`.


class SharpTask(NamespacedTask, SafeTask):
    """
    Base class for all (non-wrapper or -external) tasks in this package.
    """


class ExternalTask(NamespacedTask, luigi.ExternalTask):
    pass


class WrapperTask(NamespacedTask, luigi.WrapperTask):
    pass
