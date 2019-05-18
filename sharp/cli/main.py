from click import Group, group

from sharp.cli.config import config
from sharp.cli.scheduler import scheduler
from sharp.cli.worker import worker


# Flag when we have entered our own code.
# print("\nWelcome to the sharp CLI.\n", flush=True)
#
# --> Printing something here messes up Click bash tab completion (even with
# click.echo)


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
