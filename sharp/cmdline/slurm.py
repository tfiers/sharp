from subprocess import PIPE, run
from time import sleep

from click import argument

from sharp.cmdline.util import option, resolve_path_str, sharp_command


@sharp_command(short_help="Run sharp workers on a SLURM computing cluster.")
@argument("config_directory")
@option("-N", "--nodes", type=int, default=1)
@option("-w", "--workers-per-node", type=int, default=1)
@option("-c", "--cpus-per-worker", type=int, default=40)
@option(
    "--contact",
    default="tomas.fiers@gmail.com",
    help="Name / contact info of person responsible for SLURM job.",
)
def slurm(config_directory, nodes, workers_per_node, cpus_per_worker, contact):
    """
    Start a job on a a SLURM-managed computing cluster (https://slurm.schedmd.com/).
    The job consists of running many "sharp worker" processes in parallel.
    """
    config_dir = resolve_path_str(config_directory)
    cmd = (
        "srun",
        f"sharp worker {config_dir}",
        f"--nodes={nodes}",
        f"--ntasks-per-node={workers_per_node}",
        f"--cpus-per-task={cpus_per_worker}",
        f"--job-name=sharp ({contact})",
        f"--output={config_dir / 'logs' / 'slurm-j%j.log'}",
        f"--mem=0",
    )
    # --mem=0 means: use all available memory on a node. This also means that
    # the SLURM scheduler won't auto-kill all tasks on memory limit
    # transgression; disk swap will be used instead.
    run(cmd)
    sleep(1)
    run("squeue", stdout=PIPE, stderr=PIPE)
    print("\n")
    run(("scontrol show job", "--details"), stdout=PIPE, stderr=PIPE)
