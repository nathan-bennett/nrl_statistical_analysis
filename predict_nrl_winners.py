import matplotlib
import argparse
import mlflow
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import h2o
import numpy as np
import pyarrow.parquet as pq
from common_functions import get_metrics, log_metrics, log_artifacts, split_df, move_files, file_swap_s3
import os
import glob
import configparser
import s3fs
import sys
import random
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

    train['winner'] = train['winner'].asfactor()
    test['winner'] = test['winner'].asfactor()

    return train, test, random_categorical_list


def formulate_grid(train_set, test_set, max_run_time_secs, seed):

    # sample parameters for GBM
    gbm_params1 = {"max_depth": [i for i in range(2, 51)],
                   "ntrees": [i for i in range(2, 51)],
                   "min_rows": [i for i in range(8, 144, 8)],
                   "nbins": [16, 32, 64, 128, 256, 512, 1024],
                   "sample_rate": [i for i in np.arange(0.2, 1, 0.01)],
                   "learn_rate": [i for i in np.arange(0.01, 0.2, 0.01)],
                   "learn_rate_annealing": [i for i in np.arange(0.9, 1, 0.01)],
                   "col_sample_rate": [i for i in np.arange(0.2, 1, 0.01)],
                   "nbins_cats": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                   "col_sample_rate_per_tree": [i for i in np.arange(0.2, 1, 0.01)],
                   "min_split_improvement": [0e+00, 1e-08, 1e-06, 1e-04],
                   "distribution": ['multinomial'],
                   "histogram_type": ["UniformAdaptive",
                                      "QuantilesGlobal",
                                      "RoundRobin"],
                   "col_sample_rate_change_per_level": [i for i in np.arange(0.2, 1.5, 0.01)]}

    # Search criteria
    search_criteria = {'strategy': 'RandomDiscrete',
                       'seed': seed,
                       'max_runtime_secs': max_run_time_secs,
                       'stopping_tolerance': 0.00001}

    # Train and validate a grid of GBMs
    gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                             grid_id='gbm_grid1',
                             hyper_params=gbm_params1,
                             search_criteria=search_criteria)
    preds = ['home_team',
             'home_odds',
             'away_odds',
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

    try:
        gbm_grid.train(x=preds, y='winner',
                       training_frame=train_set,
                       validation_frame=test_set,
                       seed=seed
                       )
    except:
        h2o.cluster().shutdown()
        sys.exit()

    gbm_grid_perf = gbm_grid.get_grid(sort_by='r2', decreasing=True)

    return gbm_grid_perf, preds


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser(description='Predict NRL Scores GBM')
    args_parser.add_argument('--max_run_time_secs', type=int, required=False)
    args_parser.add_argument('--seed', type=int, required=False)
    args_parser.add_argument('--location', type=str, required=False)
    args_parser.add_argument('--config', type=str, required=False)

    args = args_parser.parse_args()

    if args.max_run_time_secs:
        max_run_time_secs = args.max_run_time_secs
    else:
        max_run_time_secs = 200

    if args.seed:
        seed = args.seed
    else:
        seed = 1234

    if args.location:
        location = args.location
    else:
        location = 'nrl-prediction-analysis.com/previous-nrl-results.parquet.gzip'

    if args.config:
        config_name = args.config
    else:
        config_name = 'nrl_config.ini'

    h2o.init()
    h2o.cluster().shutdown()

    # load in config file
    config = configparser.ConfigParser()
    config.read(config_name)

    # get secret keys from config file
    access_key = config['aws']['aws_access_key_id']
    secret_key = config['aws']['aws_secret_access_key']

    file_swap_s3(access_key, secret_key, "s3://nrl-prediction-analysis.com/mlruns/", os.getcwd() + '/mlruns/')

    mlflow.set_experiment("predict-winner-nrl")

    df = get_data(access_key, secret_key, location)

    train, test = split_df(df, random=False)

    train, test, random_categorical_list = convert_vars(train, test)

    # list of models, ordered from best to worst performing
    grid, predictors = formulate_grid(train, test, max_run_time_secs, int(seed))

    # now we log the results in mlflow
    ranking = 1
    for model in grid:

        mlflow.start_run()

        print("\nMetrics for no {} model".format(ranking))

        ranking += 1

        r2_metrics, mse_metrics, rmse_metrics, auc_metrics = get_metrics(model, train, test)

        log_metrics(r2_metrics, mse_metrics, rmse_metrics)

        script_name = sys.argv[0]

        log_artifacts(model, script_name)

        for keys in model.params:
            mlflow.log_param(keys, model.params[keys]['actual'])

        mlflow.log_param("independent_variables", predictors)

        mlflow.log_param("categorical_variables", random_categorical_list)

        # change where the artifacts save
        path = mlflow.get_artifact_uri().replace("file://", "")
        path_split = path.split('/')
        path_split[4] = "mlruns_artifacts"
        new_path = '/'.join(path_split)

        move_files(path, new_path)

        mlflow.end_run()

        # Remove files that are already saved else where
        for filename in glob.glob("gbm_grid1_model*"):
            os.remove(os.getcwd() + '/' + filename)

        if ranking == 3:
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

    h2o.cluster().shutdown()

    print("Finished")
