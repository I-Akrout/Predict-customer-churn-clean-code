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
import joblib
import numpy as np
import pandas as pd
import dataframe_image as dfi
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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
        './images/eda/Univariate_categorical_plot.png')
    
    eda_marital_income.cla()

    df_quant = df[quant_columns + ['Churn']]

    nb_churn_customer_per_age_class = df_quant.groupby(
        ['Customer_Age']).sum()['Churn']
    nb_customer_per_age_class = df_quant.groupby(
        ['Customer_Age']).count()['Churn']
    
    
    df_result = pd.DataFrame()
    df_result = 100*nb_churn_customer_per_age_class / nb_customer_per_age_class

    eda_cutomer_age = df_result.plot.bar(                       figsize=(
                           20,
                           13),
                       grid=True,
                       title='Plot of the Churn percentage per age class')
    eda_cutomer_age.set_xlabel('Customer age class')
    eda_cutomer_age.set_ylabel('Perentage of churn')
    eda_cutomer_age.figure.savefig(
        './images/eda/Univariate_quantitative_plot.png')
    
    eda_cutomer_age.cla()
    plt.figure(figsize=(20, 10))
    eda_bivariate = sns.displot(data=df, x='Total_Trans_Ct', y='Customer_Age')
    
    eda_bivariate.axes[0, 0].set_title('Bivariate plot of Customer_Age vs')


    plt.gcf().set_size_inches(15, 8)
    eda_bivariate.axes[0, 0].set_xlabel('Total_Trans_Ct')
    eda_bivariate.axes[0, 0].set_ylabel('Customer_Age')
    eda_bivariate.figure.savefig(
        './images/eda/Bivariate_plot.png')


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
            df.groupby(col).mean().Churn)
    df[response] = df[category_lst]
    df = df.replace(new_category_values)
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
    
    x_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn']
    X = pd.DataFrame()
    X[x_cols] = df[x_cols]
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

    try:
        assert len(y_train) == len(y_train_preds_lr) == len(y_train_preds_rf)
    except AssertionError as err:
        logging.error("ERROR: RF train predictions, LR train predictions and True train lables doesn't match")
        raise err

    try:
        assert len(y_train) > 0
    except AssertionError as err:
        logging.error(
            "ERROR: True train labels and predictions are empty.")
        raise err

    try:
        assert len(y_train) == len(y_train_preds_lr) == len(y_train_preds_rf)
    except AssertionError as err:
        logging.error(
            "ERROR: RF test predictions, LR test predictions and True test lables doesn't match")
        raise err

    try:
        assert len(y_train) > 0
    except AssertionError as err:
        logging.error(
            "ERROR: True test labels and predictions are empty.")
        raise err

    logging.info('INFO: Saving reports')

    logging.info('INFO: Saving random forest reports')

    logging.info('INFO: ..... Test report')
    rfc_test_df = pd.DataFrame(classification_report(
        y_test, y_test_preds_rf, output_dict=True)).transpose()
    dfi.export(rfc_test_df, './images/results/rfc_test_report.png')

    logging.info('INFO: ..... Train report')

    rfc_train_df = pd.DataFrame(
        classification_report(y_train, y_train_preds_rf, output_dict=True)).transpose()
    dfi.export(rfc_train_df, './images/results/rfc_train_report.png')

    logging.info('INFO: Saving logistic regression reports')
    logging.info('INFO: ..... Test report')
    lrc_test_df = pd.DataFrame(classification_report(
        y_test, y_test_preds_lr, output_dict=True)).transpose()
    dfi.export(lrc_test_df, './images/results/lrc_test_report.png')

    logging.info('INFO: ..... Train report')
    lrc_train_df = pd.DataFrame(
        classification_report(y_train, y_train_preds_lr, output_dict=True)).transpose()
    dfi.export(lrc_train_df, './images/results/lrc_train_report.png')

    logging.info('SUCCESS: Reports saved successfully.')



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
    try:
        assert X_train.shape[1] == X_data.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        logging.error('ERROR: Training data contains non numerical numbers')
        raise err

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


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

    try:
        assert X_train.shape[1] == X_train.select_dtypes(include=np.number).shape[1]
    except AssertionError as err:
        logging.error('ERROR: Training data contains non numerical numbers')
        raise err

    try:
        assert X_test.shape[1] == X_test.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        logging.error('ERROR: Testing data contains non numerical numbers')
        raise err

    try:
        assert pd.DataFrame(y_train).shape[0] == pd.DataFrame(y_train).select_dtypes(
            include=np.number).shape[0]
    except AssertionError as err:
        logging.error('ERROR: Training labels contains non numerical numbers')
        raise err

    try:
        assert pd.DataFrame(y_test).shape[0] == pd.DataFrame(y_test).select_dtypes(
            include=np.number).shape[0]
    except AssertionError as err:
        logging.error('ERROR: Testing labels contains non numerical numbers')
        raise err

    logging.info('INFO: Begining the training of the random forest and linear regression')
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    logging.info('INFO: Initialization of random forest parameters')
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    logging.info('INFO: Fitting data into the random forest')
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info('SUCCESS: Training random forest model finished.')


    logging.info('INFO: Fitting data into the linear regression model')
    lrc.fit(X_train, y_train)
    logging.info('SUCCESS: Training linear regression model finished.')

    logging.info('INFO: Saving the random forest model ...')
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    logging.info('SUCCESS: Random forest model saved.')

    logging.info('INFO: Saving the random forest model ...')
    joblib.dump(lrc, './models/logistic_model.pkl')
    logging.info('SUCCESS: Random forest model saved.')

    logging.info('INFO: Creating ROC curves')
    logging.info('INFO: Cleaning the plots of matplotlib')
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    logging.info('INFO: Plotting the learner regression ROC curve.')
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))

    ax = plt.gca()
    logging.info('INFO: Plotting the Random forest ROC curve.')
    plot_roc_curve(cv_rfc.best_estimator_,
                            X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.title('ROC curves of Random forest and Linear regression models')
    logging.info('INFO: Saving the figure ...')
    plt.savefig('./images/results/roc_curves.png')
    logging.info('SUCCESS: ROC curves generated and saved')

    

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
    perform_eda(df)
    X_train, X_test, Y_train, Y_test = perform_feature_engineering(df, 'Churn')
    
    train_models(X_train, X_test, Y_train, Y_test)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    y_train_preds_lr = lr_model.predict(X_train)
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)
    y_test_preds_rf = rfc_model.predict(X_test)

    classification_report_image(Y_train, 
        Y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    X_data = pd.concat([X_train, X_test])
    rfc_output_pth = './images/results/feature_importance.png'
    feature_importance_plot(rfc_model, X_data, rfc_output_pth)
