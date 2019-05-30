from abc import ABC, abstractmethod
from logging import getLogger
from traceback import format_tb
from typing import List

from luigi import Event, Target, Task
from luigi.task import Parameter, flatten
from psutil import virtual_memory

from sharp.config.load import config
from sharp.util.misc import cached


log = getLogger(__name__)


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
            dependency.complete() for dependency in self.dependencies
        ) and all(output.exists() for output in self.outputs)

    def run(self):
        if config.max_memory_usage is not None:
            self.check_memory_usage()
        self.work()
        self.complete.cache_clear()

    @abstractmethod
    def work(self):
        """
        Read input file(s) (i.e. outputs of required tasks), process the data,
        and write output file(s).
        """

    def check_memory_usage(self):
        mem_usage = virtual_memory().percent / 100
        # This is the percentage of *physical* memory used (i.e. not
        # including swap). "virtual_memory()" is thus a confusing name.
        log.info(f"Currently {mem_usage:.1%} of system memory in use.")
        if mem_usage > config.max_memory_usage:
            raise MemoryError(
                f"More than {config.max_memory_usage:.0%}% of system memory"
                f" already in use. Will not start the current task."
            )

    @property
    def dependencies(self) -> List[Task]:
        return flatten(self.requires())

    @property
    def outputs(self) -> List[Target]:
        return flatten(self.output())


@SharpTask.event_handler(Event.FAILURE)
def log_exception(task: SharpTask, exception: Exception):
    """
    Catch tasks failures and don't let Luigi swallow the useful debugging info.
    """
    log.error(
        f"Reason of task failure: {exception}\n\nTraceback:\n"
        + "".join(format_tb(exception.__traceback__))
    )


class WrapperTask(SharpTask):
    """
    A task that does no work itself, but only requires other tasks, like when
    many tasks should be triggerde, or the existence of external input files
    checked.
    """

    def work(self):
        pass


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
