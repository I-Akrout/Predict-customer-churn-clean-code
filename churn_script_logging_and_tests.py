# library doc string
'''
Description: This module contains the function test of the churn_Library.py.
This is the main testing file of the project 'Predict customer Churn
with Clean Code' as part of the Machine Learning DevOps Engineering Nanodegree Program

Author: Ismail Akrout

Date: October 2nd, 2021

Version: 0.0.1
'''
import os
import logging
import numpy as np
import pytest
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


@pytest.mark.parametrize("import_data", [cl.import_data])
def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''

    logging.info("## Testing the import data function ##")

    try:
        df_data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.mark.parametrize("perform_eda", [cl.perform_eda])
def test_eda(perform_eda):
    '''
    test perform eda function
    '''

    logging.info("## Testing the eda function ##")

    eda_pth = './images/eda'
    eda_pth_save = './images/eda_back'
    path_to_test = set(['./images/eda/Bivariate_plot.png',
                       './images/eda/Univariate_categorical_plot.png',
                        './images/eda/Univariate_quantitative_plot.png'])
    # Check if the eda folder exist

    logging.info("Checking if eda folder exists")
    folder_exist = False
    if os.path.isdir(eda_pth):
        logging.info("Folder exists.")
        logging.info("Backing up the folder.")
        os.rename(eda_pth, eda_pth_save)
        os.mkdir(eda_pth)
        folder_exist = True

    logging.info("Importing data.")
    df_data = cl.import_data("./data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    logging.info("Performing EDA ...")
    perform_eda(df_data)
    logging.info("EDA done.")

    logging.info("Starting the testing phase.")
    logging.info("Listing the content of the created folder.")
    image_lst = set([os.path.join(eda_pth, f) for f in os.listdir(eda_pth)])

    logging.info("Deleting the created folder.")
    for img_file in image_lst:
        os.remove(img_file)
    logging.info("Deleting the files: SUCCESS")
    os.rmdir(eda_pth)
    logging.info("Deleting the folder: SUCCESS")

    if folder_exist:
        logging.info("Renaming the eda folder.")
        os.rename(eda_pth_save, eda_pth)

    try:

        logging.info(
            "Testing the content of the folder and the expected files")
        assert image_lst == path_to_test

        logging.info("Generated files are conform to the expectation.")
    except AssertionError as err:
        try:

            logging.info("Testing the number of generated files")
            assert len(image_lst) == len(path_to_test)
        except AssertionError as err_1:
            logging.error('ERROR: Number of existing files is different \
                from the expected number ')
            logging.error(
                'ERROR: Number of existing files {}'.format(
                    len(path_to_test),))
            logging.error(
                'ERROR: Expected number of files {}'.format(len(image_lst),))

        logging.info("Finding the miss match in the generation.")
        set_intersection = path_to_test.intersection(image_lst)
        missing_set = path_to_test - set_intersection
        extra_set = image_lst - set_intersection

        try:
            assert len(missing_set) == 0
        except AssertionError as err_2:
            logging.error('ERROR: missing file(s) {}'.format(missing_set))
            raise err_2
        try:
            assert len(extra_set) == 0
        except AssertionError as err_3:
            logging.error('ERROR: extra file(s) {}'.format(extra_set))
            raise err_3
        raise err


@pytest.mark.parametrize("encoder_helper", [cl.encoder_helper])
def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''

    logging.info("## Testing the encoder helper function ##")

    CATEGORY_LST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    RESPONSE = [cat + '_Churn' for cat in CATEGORY_LST]

    df_data = cl.import_data("./data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = encoder_helper(df_data, CATEGORY_LST, RESPONSE)
    x_data = df_data[RESPONSE]

    try:
        assert x_data.shape[1] == x_data.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        logging.error('ERROR: Training data contains non numerical numbers')
        raise err


@pytest.mark.parametrize("perform_feature_engineering",
                         [cl.perform_feature_engineering])
def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    logging.info("## Testing the feature engineering function ##")

    CATEGORY_LST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    RESPONSE = [cat + '_Churn' for cat in CATEGORY_LST]

    df_data = cl.import_data("./data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = cl.encoder_helper(df_data, CATEGORY_LST, RESPONSE)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        df_data, 'Churn')

    total_size = df_data.shape[0]
    x_train_size = X_TRAIN.shape[0]
    x_test_size = X_TEST.shape[0]
    y_train_size = Y_TRAIN.shape[0]
    y_test_size = Y_TEST.shape[0]

    try:
        assert (x_train_size + x_test_size) == total_size
    except AssertionError as err:
        logging.error(
            'ERROR: Miss matching between x train , x test and data set size')

    try:
        assert int(
            0.7 * total_size) <= x_train_size <= (int(0.7 * total_size) + 1)
    except AssertionError as err:
        logging.error('ERROR: Train size sould be 70% of the total set size')
        raise err

    try:
        assert(y_train_size + y_test_size) == total_size
    except AssertionError as err:
        logging.error(
            'ERROR: Miss matching between y train , y test and data set size')
        raise err

    try:
        assert x_test_size == y_test_size
    except AssertionError as err:
        logging.error('ERROR: Miss matchin between x test and y test')
        raise err

    try:
        assert x_train_size == y_train_size
    except AssertionError as err:
        logging.error('ERROR: Miss matchin between x train and y train')
        raise err


@pytest.mark.parametrize("train_models", [cl.train_models])
def test_train_models(train_models):
    '''
    test train_models
    '''

    logging.info("## Testing the train model function ##")

    CATEGORY_LST = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    RESPONSE = [cat + '_Churn' for cat in CATEGORY_LST]

    df_data = cl.import_data("./data/bank_data.csv")
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    df_data = cl.encoder_helper(df_data, CATEGORY_LST, RESPONSE)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = cl.perform_feature_engineering(
        df_data, 'Churn')

    model_pth = './models/'
    model_pth_save = './models_back'
    roc_pth = './images/results/roc_curves.png'
    roc_pth_save = './images/results/roc_curves_back.png'
    models_list_content = set(
        ['./models/rfc_model.pkl', './models/logistic_model.pkl'])
    folder_exist = False

    if os.path.isdir(model_pth):
        os.rename(model_pth, model_pth_save)
        os.mkdir(model_pth)
        folder_exist = True

    roc_exist = False
    if os.path.isfile(roc_pth):
        roc_exist = True
        os.rename(roc_pth, roc_pth_save)

    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    model_list = set([os.path.join(model_pth, f)
                     for f in os.listdir(model_pth)])

    for model_file in model_list:
        os.remove(model_file)
    os.rmdir(model_pth)

    roc_file_exists = os.path.isfile(roc_pth)
    if roc_file_exists:
        os.remove(roc_pth)

    if roc_exist:
        os.rename(roc_pth_save, roc_pth)

    if folder_exist:
        os.rename(model_pth_save, model_pth)
    try:
        assert roc_file_exists
    except AssertionError as err:
        logging.error('ERROR: ROC file was not generated.')
        raise err

    set_intersection = models_list_content.intersection(model_list)
    missing_set = models_list_content - set_intersection
    extra_set = model_list - set_intersection

    try:
        assert len(missing_set) == 0
    except AssertionError as err_2:
        logging.error('ERROR: missing file(s) {}'.format(missing_set))
        raise err_2
    try:
        assert len(extra_set) == 0
    except AssertionError as err_3:
        logging.error('ERROR: extra file(s) {}'.format(extra_set))
        raise err_3


if __name__ == "__main__":
    pytest.main(args=['-s', os.path.abspath(__file__)])
