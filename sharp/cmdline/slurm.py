from subprocess import run
from time import sleep

from click import argument

from sharp.cmdline.util import option, resolve_path_str, sharp_command
from sharp.util.misc import make_parent_dirs


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
    Submit a job on a SLURM-managed computing cluster. The job consists of
    running a bunch of:
    
        sharp worker --num-subprocesses={cpus-per-worker} {CONFIG_DIRECTORY}
    
    processes in parallel. After the job is submitted, the results of "squeue"
    and "scontrol show job --details" are shown. Console output of the job
    (consolidated from all workers and subprocesses) is written to a file
    "logs/slurm-j{jobID}.log".
    """
    config_dir = resolve_path_str(config_directory)
    # fmt: off
    parallelized_script = (
        f"#!/bin/bash\n"
        f"sharp worker -n {cpus_per_worker} {config_dir}"
    )
    # fmt: on
    log_path = config_dir / "logs" / "slurm-j%j.log"
    # "%j" is substituted by slurm to the job ID.
    make_parent_dirs(log_path)
    slurm_cmd = (
        "sbatch",
        f"--nodes={nodes}",
        f"--ntasks-per-node={workers_per_node}",
        f"--cpus-per-task={cpus_per_worker}",
        f"--job-name=sharp ({contact})",
        f"--output={log_path}",
        f"--mem=0",
    )
    # --mem=0 means: use all available memory on a node. This also means that
    # the SLURM scheduler won't auto-kill all tasks on memory limit
    # transgression; disk swap will be used instead.
    run(slurm_cmd, input=parallelized_script, text=True)
    sleep(1)
    run("squeue")
    print("\n")
    run(("scontrol", "show", "job", "--details"))
