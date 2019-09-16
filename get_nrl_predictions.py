import sqlite3
from datetime import datetime
import argparse
import findspark
import configparser
from optimus import Optimus
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, FloatType
from pyspark.sql.functions import lit, row_number, collect_list, ceil, col, year, hour, when, \
    datediff, lag, sum, count, monotonically_increasing_id, date_format
from pyspark.sql.window import Window
from common_functions import name_changer, build_dict, get_record, get_time_between, send_to_db
import pandas as pd
import numpy as np
from operator import itemgetter
import h2o
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def get_data_from_db(db_path):
    conn = sqlite3.connect(db_path)

    odds = pd.read_sql_query("SELECT * FROM odds_and_fixtures", conn)

    odds.drop_duplicates(['home_team', 'away_team'], inplace=True)

    odds.rename(columns={'home_odds': 'Home Odds',
                         'away_odds': 'Away Odds',
                         'home_team': 'Home Team',
                         'away_team': 'Away Team'},
                inplace=True)

    odds['date'] = pd.to_datetime(odds['date'])
    odds['Date'] = odds['date']
    odds['Kick-off (local)'] = odds['date']
    odds['Home Score'] = np.nan
    odds['Away Score'] = np.nan
    odds['Play Off Game?'] = np.nan
    odds['Over Time?'] = np.nan

    odds = odds[['Home Team', 'Home Odds', 'Home Score',
                 'Away Team', 'Away Odds', 'Away Score',
                 'Date', 'Kick-off (local)', 'Play Off Game?', 'Over Time?']]

    previous_results = op.load.excel("http://www.aussportsbetting.com/historical_data/nrl.xlsx",
                                     skiprows=1).select(col('Home Team'), col('Home Odds'), col('Home Score'),
                                                        col('Away Team'), col('Away Odds'), col('Away Score'),
                                                        col('Date'), col('Kick-off (local)'), col('Play Off Game?'),
                                                        col('Over Time?'))

    odds_op = op.create \
        .df(pdf=odds)

    df = odds_op.union(previous_results)

    return df


def transform_df(df):
    df = df.withColumn("Home Team", name_changer_udf(col("Home Team"))) \
        .withColumn("Away Team", name_changer_udf(col("Away Team"))) \
        .withColumn("day_of_week", date_format('date', 'E')) \
        .filter(col("Play Off Game?") != "Y") \
        .withColumn("night_game", when(hour(col("Kick-off (local)")) >= 17, 1).otherwise(0)) \
        .withColumn("game_index", monotonically_increasing_id()) \
        .withColumn("extra_time", when(col("Over Time?") == "Y", lit(1)).otherwise(lit(0)))

    home_games = df.select(df["Home Team"].alias("team"),
                           df["Home Score"].alias("score"),
                           df["Home Odds"].alias("odds"),
                           df["Away Team"].alias("opp_team"),
                           df["Away Score"].alias("opp_score"),
                           df["Away Odds"].alias("opp_odds"),
                           df["Date"].alias("date"),
                           df["day_of_week"],
                           df["night_game"],
                           df["game_index"],
                           df["extra_time"],
                           df["Kick-off (local)"].alias("local_time")) \
        .withColumn("type", lit("home"))

    away_games = df.select(df["Away Team"].alias("team"),
                           df["Away Score"].alias("score"),
                           df["Away Odds"].alias("odds"),
                           df["Home Team"].alias("opp_team"),
                           df["Home Score"].alias("opp_score"),
                           df["Home Odds"].alias("opp_odds"),
                           df["Date"].alias("date"),
                           df["day_of_week"],
                           df["night_game"],
                           df["game_index"],
                           df["extra_time"],
                           df["Kick-off (local)"].alias("local_time")) \
        .withColumn("type", lit("away"))

    games = home_games.union(away_games)

    get_record_udf = udf(get_record, IntegerType())
    get_time_between_udf = udf(get_time_between, FloatType())

    games = games \
        .withColumn("year", year(col("date"))) \
        .withColumn("points_awarded",
                    when(col("score") > col("opp_score"),
                         lit(2)) \
                    .otherwise(when(col("score") < col("opp_score"),
                                    lit(0)) \
                               .otherwise(lit(1)))) \
        .sort(col("date"), col("local_time")) \
        .withColumn("rest",
                    datediff(col("date"),
                             lag(col("date"), 1) \
                             .over(Window. \
                                   partitionBy(col("team"), col("year")) \
                                   .orderBy(col("date"))))) \
        .withColumn("result", when(col("score") > col("opp_score"), lit(1)) \
                    .otherwise(when(col("score") < col("opp_score"),
                                    lit(0)) \
                               .otherwise(np.nan))) \
        .withColumn("win", when(col("score") > col("opp_score"), lit(1)) \
                    .otherwise(lit(0))) \
        .withColumn("loss", when(col("score") < col("opp_score"), lit(1)) \
                    .otherwise(lit(0))) \
        .withColumn("draw", when(col("score") == col("opp_score"), lit(1)) \
                    .otherwise(lit(0))) \
        .withColumn("record", collect_list(col("result")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("record_date", collect_list(col("date")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("record_extra_time", collect_list(col("extra_time")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("total_points", sum(col("points_awarded")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date"))) - col("points_awarded")) \
        .withColumn("total_points_after_game", sum(col("points_awarded")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("total_for", sum(col("score")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date"))) - col("score")) \
        .withColumn("total_for_after_game", sum(col("score")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("total_against", sum(col("opp_score")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date"))) - col("opp_score")) \
        .withColumn("total_against_after_game", sum(col("opp_score")) \
                    .over(Window. \
                          partitionBy(col("team"), col("year")) \
                          .orderBy(col("date")))) \
        .withColumn("total_for_per_game", col("total_for") / ((count("*")
                                                               .over(Window.
                                                                     partitionBy(col("team"), col("year"))
                                                                     .orderBy(col("date")))) - 1)) \
        .withColumn("total_against_per_game", col("total_against") / ((count("*")
                                                                       .over(Window.
                                                                             partitionBy(col("team"), col("year"))
                                                                             .orderBy(col("date")))) - 1)) \
        .withColumn("total_diff_per_game", col("total_for_per_game") - col("total_against_per_game")) \
        .withColumn("total_diff", col("total_for") - col("total_against")) \
        .withColumn("total_diff_after_game", col("total_for_after_game") - col("total_against_after_game")) \
        .withColumn("time_from_last_win",
                    get_time_between_udf(col("record"), col("record_date"), col("date"), lit("win"))) \
        .withColumn("time_from_last_extra_time_game",
                    get_time_between_udf(col("record_extra_time"), col("record_date"), col("date"), lit("win"))) \
        .withColumn("time_from_last_loss",
                    get_time_between_udf(col("record"), col("record_date"), col("date"), lit("loss"))) \
        .withColumn("time_from_last_draw", get_time_between_udf(col("record"), col("record_date"), col("date"))) \
        .withColumn("wins_in_a_row", get_record_udf(col("record"), lit(True))) \
        .withColumn("losses_in_a_row", get_record_udf(col("record"), lit(False)))

    pgames = games.toPandas().sort_values(['game_index'], ascending=False).reset_index(drop=True)

    for index, row in pgames.iterrows():
        if (index - 16) < 0 or row['date'].year != pgames.loc[(index - 16), 'date'].year:
            pgames.loc[index, 'position'] = np.nan
        else:
            table = []
            for reindex in range(index - 1, -1, -1):
                if row['date'].year != pgames.loc[reindex, 'date'].year or len(table) == 16:
                    break
                elif row['game_index'] == pgames.loc[reindex, 'game_index'] or pgames.loc[reindex, 'team'] in [i['team']
                                                                                                               for i in
                                                                                                               table]:
                    continue
                else:
                    team_pos = {}
                    team_pos['team'] = pgames.loc[reindex, 'team']
                    team_pos['points'] = pgames.loc[reindex, 'total_points_after_game']
                    team_pos['diff'] = pgames.loc[reindex, 'total_diff_after_game']
                    table.append(team_pos)
            table = sorted(table, key=itemgetter('points', 'diff'), reverse=True)
            info_by_team = build_dict(table, key='team')
            pgames.loc[index, 'position'] = info_by_team[row['team']]['index'] + 1

    for index, row in pgames.iterrows():
        if (index - 16) < 0 or row['date'].year != pgames.loc[(index - 16), 'date'].year:
            pgames.loc[index, 'position'] = np.nan
        else:
            table = []
            for reindex in range(index, -1, -1):
                if row['date'].year != pgames.loc[reindex, 'date'].year or len(table) == 16:
                    break
                elif pgames.loc[reindex, 'team'] in [i['team'] for i in table]:
                    continue
                else:
                    team_pos = {}
                    team_pos['team'] = pgames.loc[reindex, 'team']
                    team_pos['points'] = pgames.loc[reindex, 'total_points_after_game']
                    team_pos['diff'] = pgames.loc[reindex, 'total_diff_after_game']
                    table.append(team_pos)
            table = sorted(table, key=itemgetter('points', 'diff'), reverse=True)
            info_by_team = build_dict(table, key='team')
            pgames.loc[index, 'position_after_game'] = info_by_team[row['team']]['index'] + 1

    games = op.create \
        .df(pdf=pgames) \
        .withColumn("ranking_quantile",
                    when(col("position") != np.nan,
                         ceil(col("position") / 4)) \
                    .otherwise(np.nan)) \
        .withColumn("opp_position", sum(col("position")) \
                    .over(Window.partitionBy("game_index")) - col("position")) \
        .withColumn("opp_ranking_quantile",
                    when(col("opp_position") != np.nan,
                         ceil(col("opp_position") / 4)) \
                    .otherwise(np.nan)) \
        .withColumn("previous_opp_position",
                    lag(col("opp_position"), 1) \
                    .over(Window.partitionBy("team").orderBy(col("game_index")))) \
        .withColumn("previous_opp_ranking_quantile",
                    when(col("previous_opp_position") != np.nan,
                         ceil(col("previous_opp_position") / 4)) \
                    .otherwise(np.nan)) \
        .withColumn("previous_result",
                    lag(col("points_awarded"), 1) \
                    .over(Window.partitionBy("team").orderBy(col("game_index")))) \
        .withColumn("previous_result_ranking", col("previous_result") * col("previous_opp_ranking_quantile"))

    home_games = games \
        .filter(col("type") == "home") \
        .select(col("team").alias("home_team"),
                col("odds").alias("home_odds"),
                col("opp_team").alias("opp_away_team"),
                col("score").alias("home_score"),
                col("date"),
                col("local_time"),
                col("day_of_week"),
                col("night_game"),
                col("game_index"),
                col("wins_in_a_row").alias("home_wins_in_a_row"),
                col("losses_in_a_row").alias("home_losses_in_a_row"),
                col("position").alias("home_position"),
                col("ranking_quantile").alias("home_ranking_quantile"),
                col("total_points").alias("home_points"),
                col("total_for_per_game").alias("home_for_per_game"),
                col("total_against_per_game").alias("home_against_per_game"),
                col("previous_result_ranking").alias("home_previous_result"),
                col("time_from_last_win").alias("home_time_from_last_win"),
                col("time_from_last_extra_time_game").alias("home_time_from_last_extra_time_game"),
                col("time_from_last_loss").alias("home_time_from_last_loss"),
                col("time_from_last_draw").alias("home_time_from_last_draw"),
                col("rest").alias("home_rest"))

    away_games = games \
        .filter(col("type") == "away") \
        .select(col("team").alias("away_team"),
                col("odds").alias("away_odds"),
                col("opp_team").alias("opp_home_team"),
                col("score").alias("away_score"),
                col("position").alias("away_position"),
                col("total_points").alias("away_points"),
                col("wins_in_a_row").alias("away_wins_in_a_row"),
                col("losses_in_a_row").alias("away_losses_in_a_row"),
                col("ranking_quantile").alias("away_ranking_quantile"),
                col("total_for_per_game").alias("away_for_per_game"),
                col("total_against_per_game").alias("away_against_per_game"),
                col("rest").alias("away_rest"),
                col("previous_result_ranking").alias("away_previous_result"),
                col("time_from_last_win").alias("away_time_from_last_win"),
                col("time_from_last_extra_time_game").alias("away_time_from_last_extra_time_game"),
                col("time_from_last_loss").alias("away_time_from_last_loss"),
                col("time_from_last_draw").alias("away_time_from_last_draw"),
                col("date").alias("away_date"))

    df = home_games.join(away_games, (home_games['home_team'] == away_games['opp_home_team']) & \
                         (home_games['opp_away_team'] == away_games['away_team']) & \
                         (home_games['date'] == away_games['away_date'])) \
        .drop("opp_away_team", "opp_home_team", "away_date") \
        .withColumn("rest_spread", col("home_rest") - col("away_rest")) \
        .withColumn("game_id", monotonically_increasing_id()) \
        .withColumn("home_win", when(col("home_score") > col("away_score"), lit(1)) \
                    .otherwise(when(col("home_score") < col("away_score"),
                                    lit(0)) \
                               .otherwise(np.nan))) \
        .withColumn("winner", when(col("home_score") > col("away_score"), lit("home")) \
                    .otherwise(when(col("home_score") < col("away_score"),
                                    lit("away")) \
                               .otherwise("draw"))) \
        .withColumn("margin", col("home_score") - col("away_score")) \
        .withColumn("year", year(col("date"))) \
        .withColumn("hour", hour(col("local_time"))) \
        .withColumn("game_id_season", row_number().over(Window.partitionBy(col("year")).orderBy(col("date")))) \
        .withColumn("first_round", when(col("game_id_season") <= 8, lit(1)).otherwise(lit(0))) \
        .withColumn("second_round",
                    when((col("game_id_season") <= 16) & (col("game_id_season") > 8), lit(1)).otherwise(lit(0))) \
        .drop("local_time")

    pdf = df.toPandas()

    return pdf


def predict_results(df, model_path):

    probs = h2o.mojo_predict_pandas(df, model_path)

    df.reset_index(inplace=True, drop=True)

    predictions = pd.concat([df, probs], axis=1).sort_values(['date'], ascending=False)

    predictions['report_date'] = datetime.utcnow()

    print(predictions[['date', 'home_team', 'away_team', 'home', 'away', 'draw', 'home_odds', 'away_odds']].head(20))

    return predictions


def send_pandas_df_to_s3(predictions, s3_path):
    """
    :param predictions: pandas df
    :param s3_path: path to send predictions data frame
    :return: converts pandas df to spark data frame and also sends df to s3 using pyspark partitioning
    """
    predictions['start_date'] = predictions['date'].dt.date
    predictions['prediction_date'] = predictions['report_date'].dt.date

    predictions_spark = op.create \
        .df(pdf=predictions)

    predictions_spark. \
        repartition("prediction_date") \
        .write \
        .partitionBy("prediction_date") \
        .parquet(s3_path, mode='append')

    # predictions_spark. \
    #     repartition(1) \
    #     .write \
    #     .parquet(s3_path, mode='append')

    return predictions_spark


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description='Predict NRL Scores')
    args_parser.add_argument('--db_path', type=str, required=False)
    args_parser.add_argument('--table_name', type=str, required=False)
    args_parser.add_argument('--model_path', type=str, required=False)
    args_parser.add_argument('--config_name', type=str, required=False)

    args = args_parser.parse_args()
    db_path = args.db_path

    if args.db_path and args.model_path and args.table_name:
        db_path = args.db_path
        model_path = args.model_path
        table_name = args.table_name
        config_name = args.config_name
    else:
        db_path = "/home/user/nrl_analysis/nrl_stats.db"
        model_path = "mlruns_artifacts/2/6f7291973ca345b3974e2346bbdb9261/artifacts/" \
                     "gbm_grid1_model_179.zip"
        table_name = "predictions"
        config_name = 'nrl_config.ini'

    # load in config file
    config = configparser.ConfigParser()
    config.read(config_name)

    # get secret keys from config file
    access_key = config['aws']['aws_access_key_id']
    secret_key = config['aws']['aws_secret_access_key']

    # Initiate spark
    findspark.init("/home/user/spark-2.4.3-bin-hadoop2.7")
    op = Optimus()
    op.sc._jsc.hadoopConfiguration().set("fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
    op.sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", access_key)
    op.sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", secret_key)

    name_changer_udf = udf(name_changer, StringType())

    df = get_data_from_db(db_path)

    pdf = transform_df(df)

    predictions = predict_results(pdf, model_path)

    print(send_to_db(predictions, table_name, db_path))

    s3_path = "s3n://nrl-prediction-analysis.com/nrl_stats/"

    send_pandas_df_to_s3(predictions, s3_path)

    op.stop()
