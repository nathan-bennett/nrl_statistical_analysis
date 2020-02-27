from typing import Dict, Any, Union
from selenium import webdriver
import pandas as pd
from datetime import datetime, timedelta
import argparse
from common_functions import send_to_db
import numpy as np


def get_driver(path, url):
    driver = webdriver.Firefox(executable_path=path)

    driver.get(url)

    return driver


def get_teams():
    teams = [
        'Parramatta Eels',
        'Canterbury Bulldogs',
        'Canberra Raiders',
        'Gold Coast Titans',
        'NQ Cowboys',
        'Brisbane Broncos',
        'Newcastle Knights',
        'New Zealand Warriors',
        'South Sydney Rabbitohs',
        'Cronulla Sharks',
        'Penrith Panthers',
        'Sydney Roosters',
        'Manly Sea Eagles',
        'Melbourne Storm',
        'St. George Illawarra Dragons',
        'Wests Tigers',
    ]
    return teams


def get_games(driver, nrl_teams, future):
    css_selection = "#col-content > div:nth-child(8) > table:nth-child(1)"
    if future:
        css_selection = "#tournamentTable"
    raw_text = driver.find_elements_by_css_selector(css_selection)[0].text
    raw_text_split = raw_text.split("\n")[5:]
    raw_text_split = [i if i != '-' else str(1.00) for i in raw_text_split]

    games = []
    for index in range(len(raw_text_split)):
        game = dict()
        try:
            match_up_split = raw_text_split[index]
            game['time'] = datetime.strptime(match_up_split.split(' ')[0], "%H:%M")
            teams = dict()
            for nrl_team in nrl_teams:
                if nrl_team in match_up_split:
                    teams[match_up_split.index(nrl_team)] = nrl_team
            keys = [key for key in teams.keys()]
            if len(keys) == 2:
                if keys[0] < keys[1]:
                    game['home_team'] = teams[keys[0]]
                    game['away_team'] = teams[keys[1]]
                else:
                    game['home_team'] = teams[keys[1]]
                    game['away_team'] = teams[keys[0]]
            game['home_odds'] = float(raw_text_split[index + 1])
            if not future:
                scores = match_up_split.split(' ')[-1].split(":")
                game['home_score'] = float(scores[0])
                game['away_score'] = float(scores[1])
            game['draw_odds'] = float(raw_text_split[index + 2])
            game['away_odds'] = float(raw_text_split[index + 3])
            game['str_index'] = index
            games.append(game)
        except ValueError:
            continue
    return games, raw_text_split


def get_dates(raw_text_split, year):
    dates = []
    line_number = 0
    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    for line in raw_text_split:
        game_date: Dict[str, Union[Union[datetime, int], Any]] = dict()
        if year in line:
            game_date['str_date'] = line.split(year)[0] + year
            game_date['date'] = datetime.strptime(game_date['str_date'], "%d %b %Y")
            game_date['index'] = line_number
            dates.append(game_date)
        elif "Yesterday" in line:
            game_date['str_date'] = line.split(", ")[1].split(" 1 X 2")[0] + " " + str(yesterday.year)
            game_date['date'] = datetime.strptime(game_date['str_date'], "%d %b %Y")
            game_date['index'] = line_number
            dates.append(game_date)
        elif "Tomorrow" in line:
            game_date['str_date'] = line.split(", ")[1].split(" 1 X 2")[0].split(" - ")[0] + " " + str(tomorrow.year)
            game_date['date'] = datetime.strptime(game_date['str_date'], "%d %b %Y")
            game_date['index'] = line_number
            dates.append(game_date)
        elif "Today" in line:
            game_date['str_date'] = line.split(", ")[1].split(" 1 X 2")[0].split(" - ")[0] + " " + str(today.year)
            game_date['date'] = datetime.strptime(game_date['str_date'], "%d %b %Y")
            game_date['index'] = line_number
            dates.append(game_date)
        line_number += 1

    pd_dates = pd.DataFrame(dates)

    return pd_dates


def concat_games_and_dates(pd_games, pd_dates):
    for index, row in pd_games.iterrows():
        try:
            pd_games.loc[index, 'date'] = pd_dates[(pd_dates['index'] < row['str_index'])]['date'].iloc[-1]
        except KeyError:
            continue
    return pd_games


def format_games(pd_games, future):
    pd_games['game_date'] = pd_games.apply(lambda row: datetime(row['date'].year,
                                                                row['date'].month,
                                                                row['date'].day,
                                                                row['time'].hour,
                                                                row['time'].minute), axis=1)

    if future:
        pd_games['home_score'] = np.nan
        pd_games['away_score'] = np.nan

    pd_games['margin'] = pd_games['home_score'] - pd_games['away_score']

    pd_games = pd_games[['game_date',
                         'home_team',
                         'away_team',
                         'home_odds',
                         'draw_odds',
                         'away_odds',
                         'home_score',
                         'away_score',
                         'margin']]

    pd_games['scrape_date'] = datetime.now()

    return pd_games


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description='Predict NRL Scores')
    args_parser.add_argument('--path', type=str, required=False, default='/usr/local/bin/geckodriver')
    args_parser.add_argument('--table_name', type=str, required=False, default='future_odds')
    args_parser.add_argument('--db_path', type=str, required=False, default="nrl_stats.db")
    args_parser.add_argument('--year', type=str, required=False, default="2020")
    args_parser.add_argument('--future', type=bool, required=False, default=True)
    args_parser.add_argument('--url',
                             type=str,
                             required=False,
                             default='https://www.oddsportal.com/rugby-league/australia/nrl/')

    args = args_parser.parse_args()

    path = args.path
    url = args.url
    table_name = args.table_name
    future = args.future

    if future and table_name == "historical_odds":
        ValueError("Can't get future odds and then save results to historical db")

    db_path = args.db_path

    driver = get_driver(path, url)

    teams = get_teams()

    games, raw_text_split = get_games(driver, teams, future)

    pd_dates = get_dates(raw_text_split, args.year)

    pd_games = pd.DataFrame(games)

    pd_games = concat_games_and_dates(pd_games, pd_dates)

    pd_games = format_games(pd_games, future)

    print(send_to_db(pd_games, table_name, db_path))

    driver.quit()
