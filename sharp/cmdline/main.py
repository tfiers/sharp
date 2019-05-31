from click import Group, group

from sharp.cmdline.config import config
from sharp.cmdline.scheduler import scheduler
from sharp.cmdline.worker import worker


@group(
    options_metavar="<options>",
    subcommand_metavar="<command>",
    context_settings=dict(help_option_names=["-h", "--help"]),
    epilog='Type "sharp <command> -h" for more help.',
)
def main_cli():
    pass


main_cli: Group
main_cli.add_command(config)
main_cli.add_command(worker)
main_cli.add_command(scheduler)
