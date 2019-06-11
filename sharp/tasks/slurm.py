from subprocess import run
from textwrap import dedent
from time import sleep

from airflow.executors.celery_executor import app
from celery.app.control import Control
from sharp.util.misc import make_parent_dirs, resolve_path_str


JOB_NAME = "sharp_(tomas.fiers@gmail.com)"


def start_slurm_job(nodes=1, workers_per_node=1, cpus_per_worker=40):
    # fmt: off
    slurm_script = dedent(f"""
        #!/bin/bash
        srun airflow worker --concurrency={cpus_per_worker} --queues=cluster
        """
    ).strip()
    log_path = resolve_path_str("~/tomas/data/air/logs/slurm-j%j.log")
    make_parent_dirs(log_path)
    # "%j" is substituted by slurm to the job ID.
    # fmt: on
    slurm_cmd = (
        "sbatch",
        f"--nodes={nodes}",
        f"--ntasks-per-node={workers_per_node}",
        f"--cpus-per-task={cpus_per_worker}",
        f"--job-name={JOB_NAME}",
        f"--output={log_path}",
        f"--mem=0",
    )
    # --mem=0 means: use all available memory on a node. This also means that
    # the SLURM scheduler won't auto-kill all tasks on memory limit
    # transgression; disk swap will be used instead.
    run(slurm_cmd, input=slurm_script, text=True)
    sleep(1)
    run("squeue")
    print("\n")
    run(("scontrol", "show", "job", "--details"))


def cancel_slurm_job():
    celery_controller: Control = app.control
    cluster_worker_names = [f"celery@compute0{i+1}" for i in range(3)]
    celery_controller.broadcast("shutdown", destination=cluster_worker_names)
    run(("scancel", f"--jobname={JOB_NAME}"))
    # (Note inconsistency with sbatch in "jobname" argument)
    run("squeue")
