from click import Group, group

from sharp.cmdline.config import config
from sharp.cmdline.scheduler import scheduler
from sharp.cmdline.slurm import slurm
from sharp.cmdline.util import sharp_command_group
from sharp.cmdline.worker import worker


@sharp_command_group(
    context_settings=dict(help_option_names=["-h", "--help"]),
    epilog='Type "sharp <command> -h" for more help.',
)
def main_cli():
    pass


main_cli: Group
main_cli.add_command(config)
main_cli.add_command(worker)
main_cli.add_command(scheduler)
main_cli.add_command(slurm)
