import argparse
import configparser
import glob
import os
import random
import sys
from common_functions import get_metrics, log_metrics, log_artifacts, split_df, move_files, file_swap_s3
import h2o
import matplotlib
import mlflow
import pyarrow.parquet as pq
import s3fs
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from common_functions import get_metrics, log_metrics, log_artifacts, split_df, move_files

matplotlib.use('Agg')


def get_data(access_key, secret_key, location):

    # access s3 file system
    s3 = s3fs.S3FileSystem(key=access_key, secret=secret_key)

    # get parquet file from s3 as pandas data frame
    df = pq.ParquetDataset(location, filesystem=s3)\
        .read_pandas()\
        .to_pandas()

    return df


def convert_vars(train, test):

    to_categorical_list = [
        'away_rest',
        'home_rest',
        'rest_spread',
        'home_ranking_quantile',
        'away_ranking_quantile',
        'home_position',
        'away_position',
        'home_previous_result',
        'away_previous_result',
        'home_wins_in_a_row',
        'home_losses_in_a_row',
        'away_wins_in_a_row',
        'away_losses_in_a_row',
        'day_of_week',
        'hour',
    ]
    k = random.randrange(1, len(to_categorical_list)+1)

    random_categorical_list = random.sample(to_categorical_list, k)

    for col in random_categorical_list:
        train['{}'.format(col)] = train['{}'.format(col)].asfactor()
        test['{}'.format(col)] = test['{}'.format(col)].asfactor()

    return train, test, random_categorical_list


def formulate_glm(train, test):

    # set the predictors for poisson regression model
    preds = ['home_team',
             'home_odds',
             'away_odds',
             'local_time',
             'day_of_week',
             'night_game',
             'hour',
             'first_round',
             'second_round',
             'home_wins_in_a_row',
             'home_losses_in_a_row',
             'home_position',
             'home_ranking_quantile',
             'home_points',
             'home_for_per_game',
             'home_against_per_game',
             'home_rest',
             'home_previous_result',
             'home_time_from_last_win',
             'home_time_from_last_extra_time_game',
             'home_time_from_last_loss',
             'home_time_from_last_draw',
             'away_team',
             'away_position',
             'away_points',
             'away_wins_in_a_row',
             'away_losses_in_a_row',
             'away_ranking_quantile',
             'away_for_per_game',
             'away_against_per_game',
             'away_rest',
             'away_previous_result',
             'away_time_from_last_win',
             'away_time_from_last_extra_time_game',
             'away_time_from_last_loss',
             'away_time_from_last_draw',
             'rest_spread']

    multi_glm = H2OGeneralizedLinearEstimator(family='multinomial', lambda_=0,
                                              balance_classes=True,
                                              remove_collinear_columns=True,
                                              standardize=True,
                                              training_frame=test
                                              )

    multi_glm.train(x=preds,
                    y='winner',
                    training_frame=train)

    multi_mod = [multi_glm]

    return multi_mod, preds


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description='Predict NRL Scores Logistic Regression')
    args_parser.add_argument('--max_run_time_secs', type=int, required=False)
    args_parser.add_argument('--seed', type=int, required=False)
    args_parser.add_argument('--location', type=str, required=False)
    args_parser.add_argument('--config', type=str, required=False)

    args = args_parser.parse_args()

    h2o.init()

    if args.location:
        location = args.location
    else:
        location = 'nrl-prediction-analysis.com/previous-nrl-results.parquet.gzip'

    if args.config:
        config_name = args.config
    else:
        config_name = 'nrl_config.ini'

    # load in config file
    config = configparser.ConfigParser()
    config.read(config_name)

    # get secret keys from config file
    access_key = config['aws']['aws_access_key_id']
    secret_key = config['aws']['aws_secret_access_key']

    file_swap_s3(access_key, secret_key, "s3://nrl-prediction-analysis.com/mlruns/", os.getcwd() + '/mlruns/')

    mlflow.set_experiment("predict-winner-nrl-regression")

    df = get_data(access_key, secret_key, location)

    train, test = split_df(df, random=False)

    train, test, random_categorical_list = convert_vars(train, test)

    # list of models, ordered from best to worst performing
    poisson_model, predictors = formulate_glm(train, test)

    # now we log the results in mlflow
    ranking = 1
    for model in poisson_model:

        mlflow.start_run()

        print("\nMetrics for no {} model".format(ranking))

        ranking += 1

        r2_metrics, mse_metrics, rmse_metrics, auc_metrics = get_metrics(model, train, test)

        log_metrics(r2_metrics, mse_metrics, rmse_metrics)

        script_name = sys.argv[0]

        log_artifacts(model, script_name, coeff=False)

        for keys in model.params:
            mlflow.log_param(keys, model.params[keys]['actual'])

        mlflow.log_param("independent_variables", predictors)
        mlflow.log_param("categorical_variables", random_categorical_list)

        # change where the artifacts save
        path = mlflow.get_artifact_uri()
        path_split = path.split('/')
        path_split[4] = "mlruns_artifacts"
        new_path = '/'.join(path_split)

        move_files(path, new_path)

        mlflow.end_run()

        # Remove files that are already saved else where
        for filename in glob.glob("GLM*"):
            os.remove(os.getcwd() + '/' + filename)

        if ranking == 6:
            break

        mlflow.end_run()

    if os.path.isfile(os.getcwd() + "/h2o-genmodel.jar"):
        os.remove(os.getcwd() + "/h2o-genmodel.jar")

    if os.path.isfile(os.getcwd() + "/var_imp.png"):
        os.remove(os.getcwd() + "/var_imp.png")

    if os.path.isfile(os.getcwd() + "/var_imp.csv"):
        os.remove(os.getcwd() + "/var_imp.csv")

    # back up parameters, metrics and artifacts to s3
    file_swap_s3(access_key, secret_key, os.getcwd() + '/mlruns/',
                 "s3://nrl-prediction-analysis.com/mlruns/")
    file_swap_s3(access_key, secret_key, os.getcwd() + '/mlruns_artifacts/',
                 "s3://nrl-prediction-analysis.com/mlruns_artifacts/")

    print("Finished")
