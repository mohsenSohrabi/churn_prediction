import pandas as pd 
from preprocessing import delete_irrelevant_columns,encode_yes_no_columns, encode_state, \
    normalize_numerical_columns
from models import xgb_model

def main():
    # Load the train and test data from CSV files
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')
    
    # Preprocess the data
    # Delete irrelevant features that do not contribute to the prediction accuracy
    cols_for_delete = ['area_code', 'total_day_calls','total_eve_charge','total_night_charge',
                      'total_intl_charge']
    train_data, test_data = delete_irrelevant_columns(cols=cols_for_delete,
                                                      train_data=train_data,
                                                      test_data=test_data)
    
    # Encode columns with 'yes' and 'no' values using label encoding
    cols_for_label_encoding = ['international_plan', 'voice_mail_plan','churn']
    train_data,test_data = encode_yes_no_columns(cols= cols_for_label_encoding,
                                                 train_data= train_data,
                                                 test_data= test_data)
    
    # Encode the 'state' column using one-hot encoding
    train_data ,test_data = encode_state(train_data=train_data,
                                         test_data=test_data)
    
    # Normalize numerical columns using MinMaxScaler
    train_data, test_data = normalize_numerical_columns(train_data=train_data,
                                                        test_data=test_data)
    
    # End of preprocessing

    # Prepare the data for feeding to the models
    X_train = train_data.drop(columns=['churn'])
    y_train = train_data['churn']

    X_test_ids = test_data['id']
    X_test = test_data.drop(columns='id')

    # Use the trained model to make predictions on the test data
    predictions = xgb_model(X_train=X_train,y_train=y_train,X_test=X_test)
    
    # Create a submission DataFrame with the "id" and "churn" columns
    xgb_submission = pd.DataFrame({'id': X_test_ids, 'churn': predictions})

    # Save the submission DataFrame to a CSV file
    xgb_submission.to_csv('submissions/xgb_submission.csv', index=False)

if __name__ == "__main__":
    main()
