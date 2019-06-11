from datetime import datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sharp.config.default.raw_data import flat_recordings_list
from sharp.tasks import downsample


# 1. Airflow will only run tasks with a start_date in the past.
# 2. Specifying the start date in default_args is not supported when using the
#    "dag >> task[s]" syntax.
default_operator_kwargs = dict(owner="Tomas")
dag = DAG(
    "papermaker",
    start_date=datetime(1994, 1, 1),
    default_args=default_operator_kwargs,
    schedule_interval=None,
)


def as_ID(text: str):
    for forbidden_char in ", ()":
        text = text.replace(forbidden_char, "_")
    return text


for file_ID in flat_recordings_list:
    dag >> PythonOperator(
        python_callable=downsample,
        op_args=[file_ID],
        task_id=as_ID(f"Downsample {file_ID.short_str}"),
    )
