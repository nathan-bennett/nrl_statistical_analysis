import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import argparse
from common_functions import send_to_db


def get_data(db_path):

    conn = sqlite3.connect(db_path)

    odds = pd.read_sql_query("SELECT * FROM odds", conn)
    future_games = pd.read_sql_query("SELECT * FROM next_fixture", conn)

    return odds, future_games


def join_odds_and_fixtures(odds, future_games):

    two_weeks_ago = datetime.utcnow() - timedelta(days=14)

    odds['scrape_date'] = pd.to_datetime(odds['scrape_date'])

    latest_odds = odds[(odds['scrape_date'] >= two_weeks_ago)]\
        .sort_values(['scrape_date'])

    latest_odds['home_odds'] = latest_odds['home_odds'].astype(float)
    latest_odds['away_odds'] = latest_odds['away_odds'].astype(float)

    # As well as getting the latest odds, we will also get the average odds for each game
    mean_odds = pd.DataFrame(latest_odds.groupby(['home_team', 'away_team'])[['home_odds', 'away_odds']].mean())

    mean_odds.columns = ['mean_home_odds', 'mean_away_odds']

    nrl_odds = pd.merge(latest_odds.drop_duplicates(['home_team', 'away_team'], keep='last'),
                        future_games,
                        how='inner',
                        left_on=['home_team', 'away_team'],
                        right_on=['home_team', 'away_team']).merge(mean_odds,
                                                                   how='inner',
                                                                   left_on=['home_team', 'away_team'],
                                                                   right_on=['home_team', 'away_team']
                                                                   )
    nrl_odds.drop_duplicates(['home_team', 'away_team', 'date'],
                             inplace=True)

    nrl_odds['report_date'] = datetime.utcnow()

    return nrl_odds


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description='Predict NRL Scores')
    args_parser.add_argument('--db_path', type=str, required=True)
    args_parser.add_argument('--table_name', type=str, required=False)

    args = args_parser.parse_args()
    db_path = args.db_path

    if args.table_name:
        table_name = args.table_name
    else:
        table_name = 'odds_and_fixtures'

    odds, future_games = get_data(db_path)

    games = join_odds_and_fixtures(odds, future_games)

    print(send_to_db(games, table_name, db_path))
