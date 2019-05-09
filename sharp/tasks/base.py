from abc import ABC
from typing import List

from luigi import Target, Task
from luigi.task import Parameter, flatten

from sharp.config.load import config
from sharp.util.misc import cached


class SharpTask(Task, ABC):
    """
    Base class for all tasks in this package.

    A luigi Task in which `output()` is only ever executed by luigi if all
    dependencies specified in `requires()` have been completed.
    """

    config_id = Parameter(default=config.config_id)
    # Add a constant "parameter" to each task, so the Luigi task scheduler is
    # able to distinguish between tasks started from different python
    # processes, each with a different `config.py` file.
    # (This `default=` trick is an ugly but necessary luigi hack).

    # Caching this function avoids many spurious task completion checks. It
    # requires invalidation of a task's cache after the task has run,
    # though.
    @cached
    def complete(self) -> bool:
        """
        Whether this tasks's output exists, and its dependencies have
        completed.
        """
        return all(
            dependency.complete() for dependency in self._dependencies
        ) and all(output.exists() for output in self._outputs)

    def run(self):
        self.work()
        self.complete.cache_clear()

    def work(self):
        pass

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

    work = None


class CustomParameter(Parameter):
    def _warn_on_wrong_param_type(self, param_name, param_value):
        """ Don't check param type. """


class TaskParameter(CustomParameter):
    """
    An instantiated task. (Not like luigi.TaskParameter, which should really
    be named "TaskTypeParameter").

    Only to be used programatically. (We do not implement the parse() method to
    convert a string from the command line to an instance).
    """


class TaskListParameter(CustomParameter):
    """ A sequence of instantiated tasks """
