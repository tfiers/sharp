from typing import List

from luigi import Target, Task
from luigi.task import Parameter, flatten
from sharp.config.load import config
from sharp.util.misc import cached


class SharpTask(Task):
    """
    Base class for all tasks in this package.

    A luigi Task in which `output()` is only ever executed by luigi if all
    dependencies specified in `requires()` have been completed.
    """

    config_id = Parameter(default=config.config_id)
    # (This `default` trick is an ugly but necessary luigi hack).

    @cached
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


class CustomParameter(Parameter):
    def _warn_on_wrong_param_type(self, param_name, param_value):
        """ Don't check param type. """


class TaskParameter(CustomParameter):
    """
    An instantiated task. (Not like luigi.TaskParameter, which should really
    be named "TaskTypeParameter").

    Only to be used programatically. (We do not implement the parse() method to
    convert a string to an instance).
    """


class TaskListParameter(CustomParameter):
    """ A sequence of instantiated tasks """
