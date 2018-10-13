from typing import List

from luigi import Target, Task
from luigi.task import Config, Parameter, flatten


class SharpConfig(Config):
    config_id = Parameter()
    # Setting this to a custom value allows to run multiple pipelines (each
    # with a different `luigi.toml` config file) in parallel. Each such
    # pipeline / config file corresponds to a different `config_id`.


class SharpTask(Task):
    """
    Base class for all tasks in this package.

    A luigi Task in which `output()` is only ever executed by luigi if all
    dependencies specified in `requires()` have been completed.
    """

    config_id = Parameter(default=SharpConfig().config_id)
    # (This `default` trick is an ugly luigi hack).

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


class ExternalTask(SharpTask):
    """
    Makes sure external dependencies exist.
    """

    run = None
