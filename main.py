import pandas as pd 
from preprocessing import delete_irrelevant_columns,encode_yes_no_columns, encode_state, normalize_numerical_columns

def main():

    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')
    
    # preprocessing data
    # delete irrelevant features that does not help to have a better prediction
    cols_for_delete = ['area_code', 'total_day_calls','total_eve_charge','total_night_charge',
                      'total_intl_charge']
    train_data, test_data = delete_irrelevant_columns(cols=cols_for_delete,
                                                      train_data=train_data,
                                                      test_data=test_data)
    
    # using label encoding for two columns that include yes/no values
    cols_for_label_encoding = ['international_plan', 'voice_mail_plan']
    train_data,test_data = encode_yes_no_columns(cols= cols_for_label_encoding,
                                                 train_data= train_data,
                                                 test_data= test_data)
    # using one-hot encoding for state
    train_data ,test_data = encode_state(train_data=train_data,
                                         test_data=test_data)
    
    # Normalizing numerical columns
    train_data, test_data = normalize_numerical_columns(train_data=train_data,
                                                        test_data=test_data)
    print(train_data.shape, test_data.shape)
    print(set(test_data.columns).difference(train_data.columns))
if __name__ == "__main__":
    main()



