from datetime import datetime, timedelta

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator



def dag_t1(**kwargs):

    print('dag_t1: started')
    print('dag_t1: {}'.format(kwargs))
    print('dag_t1: completed')
    return



def dag_t2(**kwargs):

    print('dag_t2: started')
    print('dag_t2: {}'.format(kwargs))
    print('dag_t2: completed')
    return



def dag_t3(**kwargs):

    print('dag_t3: started')
    print('dag_t3: {}'.format(kwargs))
    print('dag_t3: completed')
    return



with DAG(
    'tutorial_airflow',
    default_args={
        'depends_on_past': False,
        'email': ['tanweekek@gmail.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='A simple tutorial DAG',

    # ┌───────────── minute (0–59)
    # │ ┌───────────── hour (0–23)
    # │ │ ┌───────────── day of the month (1–31)
    # │ │ │ ┌───────────── month (1–12)
    # │ │ │ │ ┌───────────── day of the week (0–6) (Sunday to Saturday;
    # │ │ │ │ │                                   7 is also Sunday on some systems)
    # │ │ │ │ │
    # │ │ │ │ │
    # * * * * * <command to execute>
    schedule_interval='*/5 * * * *',

    start_date=datetime(2024, 2, 28),
    dagrun_timeout=timedelta(seconds=5),
    catchup=False,
    tags=["tutorial"],
) as dag:
    # define tasks by instantiating operators
    t1 = PythonOperator(
        task_id='t1',
        python_callable=dag_t1,
        op_kwargs={'arg1': 1, 'arg2': 2}
    )

    t2 = PythonOperator(
        task_id='t2',
        python_callable=dag_t2
    )

    t3 = PythonOperator(
        task_id='t3',
        python_callable=dag_t3
    )

    t1 >> [t2, t3]
