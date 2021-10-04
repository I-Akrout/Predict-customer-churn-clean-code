# library doc string
'''
Description: This module containes and optimized, consice and clean code equivalent
to churn_notebook.ipynb. This is the main file of the project 'Predict customer Churn
with Clean Code' as part of the Machine Learning DevOps Engineering Nanodegree Program

Author: Ismail Akrout

Date: October 2nd, 2021

Version: 0.0.1
'''

# import libraries
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

now = datetime.now
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info('SUCCESS: file {} loaded successfully'.format(pth,))
    except FileNotFoundError as err:
        logging.error('ERROR: file {} not found'.format(pth,))
        raise err
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            'ERROR: argument df in perform_eda is expected to be {} but is an instance of {}'.format(
                pd.DataFrame, type(df)))
        raise err

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]
    column_name = set(cat_columns + quant_columns)
    try:
        df_columns = set(df.columns)
        assert column_name <= df_columns
    except AssertionError as err:
        logging.error('ERROR: Missing column names {}.'.format(
            column_name - column_name.intersection(df_columns)))
        raise err

    df_cat = df[cat_columns + ['Churn']]
    df_selected_feature = df_cat.groupby(
        ['Income_Category', 'Marital_Status']).sum()['Churn']
    df_selected_feature = df_selected_feature.reset_index()
    df_selected_feature = df_selected_feature.set_index('Income_Category')

    df_result = pd.DataFrame()
    marital_status_list = df_selected_feature['Marital_Status'].unique()

    for marital_status in marital_status_list:
        df_result[marital_status] = df_selected_feature.loc[df_selected_feature['Marital_Status']
                                                            == marital_status]['Churn']

    eda_marital_income = df_result.plot.bar(
        stacked=True,
        figsize=(
            20,
            13),
        grid=True,
        title='Plot of the number of leaving users per income class and per Marital_Status')
    eda_marital_income.set_xlabel('Income Category')
    eda_marital_income.set_ylabel('Number of leaving customers')
    eda_marital_income.figure.savefig(
        './images/eda/Univariate_quantitative_plot.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            'ERROR: argument df in encoder_helper is expected to be {} but is an instance of {}'.format(
                pd.DataFrame, type(df)))
        raise err

    try:
        assert all(isinstance(elm, str) for elm in category_lst)
    except AssertionError as err:
        logging.error(
            'ERROR: All the element in category_list should be of type str')
        raise err

    try:
        assert all(isinstance(elm, str) for elm in response)
    except AssertionError as err:
        logging.error(
            'ERROR: All the element in response should be of type str')
        raise err

    try:
        assert len(response) == len(category_lst)
    except AssertionError as err:
        logging.error(
            'ERROR: category_lst and response should have the same length')
        raise err

    new_category_values = {}
    logging.info('INFO : Transforming the categorical data')
    for idx, col in enumerate(category_lst):
        new_category_values[response[idx]] = dict(
            df.groupby('Gender').mean().Churn)
    df[response] = df[category_lst]
    df.replace(new_category_values)
    logging.info('SUCCESS: Categorical data transformation finished.')

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error(
            'ERROR: argument df in perform_feature_engineering is expected to be {} but is an instance of {}'.format(
                pd.DataFrame, type(df)))
        raise err

    try:
        assert isinstance(response, str)
    except AssertionError as err:
        logging.error(
            'ERROR: argument response in perform_feature_engineering is expected to be {} but is an instance of {}'.format(
                str,
                type(response)))
        raise err
    logging.info('INFO: Splitting data into train and test (70%, 30%).')
    X = df.loc[:, ~df.columns.isin(
        ['Churn', 'Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'])]
    Y = df['Churn']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)
    logging.info('SUCCESS: Data splitting finished.')
    logging.info('INFO: X_train size {}.'.format(X_train.shape,))
    logging.info('INFO: X_test size {}.'.format(X_test.shape,))
    logging.info('INFO: Y_train size {}.'.format(Y_train.shape,))
    logging.info('INFO: Y_test size {}.'.format(Y_test.shape,))
    return X_train, X_test, Y_train, Y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == "__main__":
    path_to_data = "./data/bank_data.csv"
    df = import_data(path_to_data)
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    response = [cat + '_Churn' for cat in category_lst]
    df = encoder_helper(df, category_lst, response)
    df.head()
    d = perform_feature_engineering(df, 'Churn')

    perform_eda(df)
