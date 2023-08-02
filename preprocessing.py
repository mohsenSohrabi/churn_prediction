import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def delete_irrelevant_columns(cols, train_data,test_data):
    """
    This function deletes the specified columns from the train and test data.
    
    :param cols: list of column names to be deleted
    :param train_data: DataFrame containing the training data
    :param test_data: DataFrame containing the test data
    :return: updated train and test data with the specified columns deleted
    """
    # drop the specified columns from the train and test data
    train_data = train_data.drop(columns=cols)
    test_data = test_data.drop(columns=cols)
    
    return train_data, test_data


def encode_yes_no_columns(cols , train_data, test_data):
    """
    This function encodes columns with 'yes' and 'no' values in the train and test data.
    
    :param cols: list of column names to be encoded
    :param train_data: DataFrame containing the training data
    :param test_data: DataFrame containing the test data
    :return: updated train and test data with the specified columns encoded
    """
    # create a mapping dictionary to map 'yes' and 'no' values to 1 and 0 respectively
    mapping = {'yes': 1, 'no': 0}
    
    # apply the mapping to the specified columns in the train and test data
    train_data[cols] = train_data[cols].applymap(mapping.get).astype(int)
    # test data does not include 'churn' and we need to exclude it
    cols = [col for col in cols if col != 'churn']
    test_data[cols] = test_data[cols].applymap(mapping.get).astype(int)
    
    return train_data, test_data


def encode_state(train_data, test_data):
    """
    This function encodes the 'state' column in the train and test data using one-hot encoding.
    
    :param train_data: DataFrame containing the training data
    :param test_data: DataFrame containing the test data
    :return: updated train and test data with the 'state' column encoded
    """
    # use one-hot encoding to encode the 'state' column in the train and test data
    train_data = pd.get_dummies(train_data, columns=['state'])
    test_data = pd.get_dummies(test_data, columns=['state'])
    
    return train_data, test_data



def normalize_numerical_columns(train_data, test_data):
    """
    This function normalizes numerical columns in the train and test data using MinMaxScaler.
    
    :param train_data: DataFrame containing the training data
    :param test_data: DataFrame containing the test data
    :return: updated train and test data with numerical columns normalized
    """
    # select numerical columns from the training data
    numerical_columns = train_data.select_dtypes(include=['int64','float64']).columns
    
    # create a MinMaxScaler object
    scaler = MinMaxScaler()

    # fit the scaler to the numerical columns of the training data
    scaler.fit(train_data[numerical_columns])

    # transform the numerical columns of the train and test data
    train_data[numerical_columns] = scaler.transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    
    return train_data, test_data