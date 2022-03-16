import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_telco(telco_df):
    '''
    This functions take telco df and cleans it. Makes categorical variables into numerical.
    Returns a clean df. 
    '''
    # drop duplicate columns
    telco_df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    # drop null values stored as whitespace
    telco_df.total_charges = telco_df.total_charges.str.strip()
    telco_df = telco_df[telco_df.total_charges != '']
    # convert total charges to float
    telco_df['total_charges'] = telco_df.total_charges.astype(float)
    # convert binary categorical variables to numerical
    telco_df['gender_encoded'] = telco_df.gender.map({'Female': 1, 'Male': 0})
    telco_df['partner_encoded'] = telco_df.partner.map({'Yes': 1, 'No': 0})
    telco_df['dependents_encoded'] = telco_df.dependents.map({'Yes': 1, 'No': 0})
    telco_df['phone_service_encoded'] = telco_df.phone_service.map({'Yes': 1, 'No': 0})
    telco_df['paperless_billing_encoded'] = telco_df.paperless_billing.map({'Yes': 1, 'No': 0})
    telco_df['churn_encoded'] = telco_df.churn.map({'Yes': 1, 'No': 0})
    # get dummies for categorical variables
    dummy_df = pd.get_dummies(telco_df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    # concat dummy variables to clean df
    telco_df = pd.concat([telco_df, dummy_df], axis=1)
    # return clean data
    return telco_df

def split_telco_data(telco_df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test.
    '''
    train_validate, test = train_test_split(telco_df, test_size=.2, 
                                        random_state=177, 
                                        stratify=telco_df.churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=177, 
                                   stratify=train_validate.churn)
    return train, validate, test

