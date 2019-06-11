from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sharp.config.default.raw_data import flat_recordings_list
from sharp.tasks.downsample import downsample
from sharp.tasks.slurm import start_slurm_job, cancel_slurm_job
from sharp.util.misc import as_ID


PAST = datetime(1994, 1, 1)
# Airflow will only run tasks with a start_date in the past.

dag = DAG("papermaker", start_date=PAST, schedule_interval=None)

start_workers = PythonOperator(
    python_callable=start_slurm_job, task_id="start_slurm_job"
)

stop_workers = PythonOperator(
    python_callable=cancel_slurm_job, task_id="cancel_slurm_job"
)

dag >> start_workers

for file_ID in flat_recordings_list:
    downsample_task = PythonOperator(
        python_callable=downsample,
        op_args=[file_ID],
        task_id=as_ID(f"Downsample {file_ID.short_str}"),
        queue="cluster",
    )
    start_workers >> downsample_task >> stop_workers
