import xgboost as xgb
import json

def xgb_model(X_train, y_train, X_test):
    """
    This function trains an XGBoost model on the input train data and makes predictions on the input test data.
    
    :param X_train: DataFrame containing the training features
    :param y_train: Series containing the training labels
    :param X_test: DataFrame containing the test features
    :return: array of predictions made by the trained XGBoost model on the test data
    """
    # load the XGBoost parameters from the JSON file
    with open('config/xgb_best_params.json', 'r') as f:
        xgb_params = json.load(f)

    # create a DMatrix object for the train data
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # create a DMatrix object for the test data
    dtest = xgb.DMatrix(X_test)

    # train an XGBoost model on the train data
    model = xgb.train(xgb_params, dtrain)

    # make predictions on the test data
    predictions = model.predict(dtest)
    
    return predictions
