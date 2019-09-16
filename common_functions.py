import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import h2o
import shutil
import sqlite3
import math


def split_df(df, random=True, split=0.75, seed=1234):
    # usually I split the data frame into a random 75/25 split, sometimes I split it based on date
    # depending if we are predicting values into the future

    h2o.init()

    if random:
        hf = h2o.H2OFrame(df)

        splits = hf.split_frame([split], seed=seed)

        train_hf = splits[0]
        test_hf = splits[1]
    else:
        df.sort_values(['date'], inplace=True)
        train_index = [i for i in range(int(len(df) * 75 / 100) + 1)]

        train = df.iloc[df.index.isin(list(train_index))]
        test = df.iloc[~df.index.isin(list(train_index))]

        train_hf = h2o.H2OFrame(train)
        test_hf = h2o.H2OFrame(test)

    return train_hf, test_hf


def send_to_db(games, table_name, db_name):

    conn = sqlite3.connect(db_name)

    games.to_sql(table_name,
                 conn,
                 if_exists='append',
                 index=False)

    return "Finished"


def get_metrics(h2o_model, train, valid, test=False, logistic_mod=False, gbm_mod=False):
    """
    @param h2o_model: pandas data frame to save as csv
    @param train: h2o train set
    @param valid: h2o validation set
    @param test: h2o test set
    @param logistic_mod: is it a logistic model
    @param gbm_mod: is it a gbm model
    @return: multiple lists of metrics
    """

    if logistic_mod:
        if h2o_model.family == 'binomial':
            is_bernoulli = True
        else:
            is_bernoulli = False
    else:
        if gbm_mod:
            if h2o_model.params['distribution']['actual'] == 'bernoulli':
                is_bernoulli = True
            else:
                is_bernoulli = False
        else:
            is_bernoulli = False

    print("Is this bernoulli: {}".format(is_bernoulli))

    if test is not False:
        train_r2 = h2o_model.model_performance(test_data=train).r2()
        validation_r2 = h2o_model.model_performance(test_data=valid).r2()
        test_r2 = h2o_model.model_performance(test_data=test).r2()

        train_mse = h2o_model.model_performance(test_data=train).mse()
        validation_mse = h2o_model.model_performance(test_data=valid).mse()
        test_mse = h2o_model.model_performance(test_data=test).mse()

        train_rmse = h2o_model.model_performance(test_data=train).rmse()
        validation_rmse = h2o_model.model_performance(test_data=valid).rmse()
        test_rmse = h2o_model.model_performance(test_data=test).rmse()

        r2_metrics = [train_r2, validation_r2, test_r2]
        mse_metrics = [train_mse, validation_mse, test_mse]
        rmse_metrics = [train_rmse, validation_rmse, test_rmse]

        if is_bernoulli:
            train_auc = h2o_model.model_performance(test_data=train).auc()
            validation_auc = h2o_model.model_performance(test_data=valid).auc()
            test_auc = h2o_model.model_performance(test_data=test).auc()

            auc_metrics = [train_auc, validation_auc, test_auc]

            print("\nTrain Metrics:\nR2: {}\nMSE: {}\nRMSE: {}\nAUC: {}\n"
                  .format(train_r2, train_mse, train_rmse, train_auc))

            print("Validation Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\nAUC: {}\n"
                  .format(validation_r2, validation_mse, validation_rmse, validation_auc))

            print("Test Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\nAUC: {}\n"
                  .format(test_r2, test_mse, test_rmse, test_auc))
        else:
            auc_metrics = []

            print("\nTrain Metrics:\nR2: {}\nMSE: {}\nRMSE: {}\n"
                  .format(train_r2, train_mse, train_rmse))

            print("Validation Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\n"
                  .format(validation_r2, validation_mse, validation_rmse))

            print("Test Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\n"
                  .format(test_r2, test_mse, test_rmse))
    else:
        train_r2 = h2o_model.model_performance(test_data=train).r2()
        validation_r2 = h2o_model.model_performance(test_data=valid).r2()

        train_mse = h2o_model.model_performance(test_data=train).mse()
        validation_mse = h2o_model.model_performance(test_data=valid).mse()

        train_rmse = h2o_model.model_performance(test_data=train).rmse()
        validation_rmse = h2o_model.model_performance(test_data=valid).rmse()

        r2_metrics = [train_r2, validation_r2]
        mse_metrics = [train_mse, validation_mse]
        rmse_metrics = [train_rmse, validation_rmse]

        if is_bernoulli:
            train_auc = h2o_model.model_performance(test_data=train).auc()
            validation_auc = h2o_model.model_performance(test_data=valid).auc()

            auc_metrics = [train_auc, validation_auc]

            print("\nTrain Metrics:\nR2: {}\nMSE: {}\nRMSE: {}\nAUC: {}\n"
                  .format(train_r2, train_mse, train_rmse, train_auc))

            print("Validation Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\nAUC: {}\n"
                  .format(validation_r2, validation_mse, validation_rmse, validation_auc))
        else:
            auc_metrics = []

            print("\nTrain Metrics:\nR2: {}\nMSE: {}\nRMSE: {}\n"
                  .format(train_r2, train_mse, train_rmse))

            print("Validation Metrics: \nR2: {}\nMSE: {}\nRMSE: {}\n"
                  .format(validation_r2, validation_mse, validation_rmse))

    return r2_metrics, mse_metrics, rmse_metrics, auc_metrics


def log_metrics(r2_metrics, mse_metrics, rmse_metrics, auc_metrics=[]):
    mlflow.log_metric("MSE_train", mse_metrics[0])
    mlflow.log_metric("RMSE_train", rmse_metrics[0])
    mlflow.log_metric("R2_train", r2_metrics[0])

    mlflow.log_metric("MSE_validation", mse_metrics[1])
    mlflow.log_metric("RMSE_validation", rmse_metrics[1])
    mlflow.log_metric("R2_validation", r2_metrics[1])

    try:
        mlflow.log_metric("MSE_test", mse_metrics[2])
        mlflow.log_metric("RMSE_test", rmse_metrics[2])
        mlflow.log_metric("R2_test", r2_metrics[2])
    except IndexError:
        pass

    # This would get used if we were measuring a binomial variable
    try:
        mlflow.log_metric("AUC_train", auc_metrics[0])
        mlflow.log_metric("AUC_validation", auc_metrics[1])
        try:
            mlflow.log_metric("AUC_test", auc_metrics[2])
        except IndexError:
            pass
    except IndexError:
        pass

    return "Finished logging metrics"


def log_artifacts(grid_model, script, coeff=False):
    # check if jar file exists, if it does then delete it as it will already be backed up in mlruns folder
    if os.path.isfile(os.getcwd() + "/h2o-genmodel.jar"):
        os.remove(os.getcwd() + "/h2o-genmodel.jar")

    if os.path.isfile(os.getcwd() + "/var_imp.png"):
        os.remove(os.getcwd() + "/var_imp.png")

    if os.path.isfile(os.getcwd() + "/var_imp.csv"):
        os.remove(os.getcwd() + "/var_imp.csv")

    model_python = h2o.save_model(model=grid_model, path=os.getcwd(), force=True)
    # mlflow.log_artifact(model_python)
    mlflow.log_artifact(model_python)

    mojo_path = grid_model.download_mojo(path=os.getcwd(),
                                         get_genmodel_jar=True)

    grid_model.varimp_plot(server=True)
    plt.savefig('var_imp.png', bbox_inches='tight')

    var_imps = pd.DataFrame(grid_model.varimp(),
                            columns=['name', 'relative_importance', 'scaled_importance', 'percentage'])

    var_imps.to_csv("var_imp.csv", index=False)

    # Log mojo file and jar file
    mlflow.log_artifact(mojo_path)

    mlflow.log_artifact(os.getcwd() + "/h2o-genmodel.jar")
    mlflow.log_artifact(os.getcwd() + "/var_imp.png")
    mlflow.log_artifact(os.getcwd() + "/var_imp.csv")
    mlflow.log_artifact(os.getcwd() + "/" + script)

    if coeff:
        if os.path.isfile(os.getcwd() + "/coefficients-{}.csv".format(datetime.utcnow().date())):
            os.remove(os.getcwd() + "/coefficients-{}.csv".format(datetime.utcnow().date()))
        grid_model._model_json['output']['coefficients_table'].as_data_frame().to_csv("coefficients-{}.csv"
                                                                                      .format(datetime.utcnow().date()))
        mlflow.log_artifact(os.getcwd() + "/coefficients-{}.csv".format(datetime.utcnow().date()))

    return "Finished logging artifacts"


def move_files(source, destination):
    files = os.listdir(source)

    if not os.path.exists(destination):
        os.makedirs(destination)

    for file in files:
        shutil.move(source+'/'+file, destination)

    return "Finished moving files"


def file_swap_s3(access_key, secret_key, source, destination):

    return os.system("AWS_ACCESS_KEY_ID={} "
                     "AWS_SECRET_ACCESS_KEY={} "
                     "aws s3 sync {} {}".format(access_key, secret_key, source, destination))


# The dataset contained different names for different teams,
# so the code below simply changes the teams with multiple names to a common name
def name_changer(a):
    if a == "St George Dragons" or a == "St George Illawarra Dragons":
        return 'St. George Illawarra Dragons'
    elif a == 'Canterbury Bulldogs':
        return 'Canterbury-Bankstown Bulldogs'
    elif a == 'Manly Warringah Sea Eagles' or a == 'Manly Sea Eagles':
        return 'Manly-Warringah Sea Eagles'
    elif a == 'Cronulla Sharks' or a == 'Cronulla Sutherland Sharks':
        return 'Cronulla-Sutherland Sharks'
    elif a == 'North QLD Cowboys':
        return 'North Queensland Cowboys'
    else:
        return a


def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def get_record(results, wins=True):
    if wins:
        lookup = 1
    else:
        lookup = 0
    record_tracker = 0
    if len(results) > 1:
        results = results[:-1]  # so we do not include the current game
        record_tracker_index = len(results)-1
        while record_tracker_index >= 0 and results[record_tracker_index] == lookup:
            record_tracker += 1
            record_tracker_index -= 1
    else:
        record_tracker = np.nan
    return record_tracker


def get_time_between(rec, rec_date, d8, game_result="draw"):
    if game_result == "win":
        lookup = 1
    elif game_result == "loss":
        lookup = 0
    else:
        lookup = 0.5
    rec = [0.5 if math.isnan(re) else re for re in rec][:-1][::-1]
    rec_date = rec_date[:-1][::-1]
    if lookup in rec:
        rec_index = rec.index(lookup)
        last_d8 = rec_date[rec_index]
        time_between = (d8 - last_d8).total_seconds()/86400
    else:
        time_between = np.nan
    return time_between
