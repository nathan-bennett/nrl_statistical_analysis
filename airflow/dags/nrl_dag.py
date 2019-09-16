import datetime as dt

from airflow import DAG
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'nathan',
    'start_date': dt.datetime(2019, 8, 12),
    'concurrency': 1,
    'retries': 1
}

db_path = '/home/user/nrl_analysis/nrl_stats.db'

with DAG('nrl_dag',
         catchup=False,
         default_args=default_args,
         schedule_interval='0 */4 * * *'
         ) as dag:
    get_odds = BashOperator(task_id='get_odds',
                            bash_command='python3 /home/user/nrl_analysis/nrl_odds_scraping.py --db_path {}'
                            .format(db_path))
    get_next_games = BashOperator(task_id='get_next_games',
                                  bash_command='python3 /home/user/nrl_analysis/get_next_games.py --db_path {}'
                                  .format(db_path))

    concat_odds_and_games = BashOperator(task_id='concat_odds_and_games',
                                         bash_command='python3 /home/user/nrl_analysis/concat_odds_and_fixtures.py '
                                                      '--db_path {}'
                                         .format(db_path))

    get_predictions = BashOperator(task_id='get_predictions',
                                   bash_command='python3 /home/user/nrl_analysis/get_nrl_predictions.py --db_path {}'
                                   .format(db_path))

get_odds >> get_next_games >> concat_odds_and_games >> get_predictions
