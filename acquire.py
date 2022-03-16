from importlib import import_module
import pandas as pd
from pydataset import data
from env import get_db_url
import os

def get_telco_data(use_cache=True):
    '''
    This Function reads the telco data from Codeup Database and writes ir to a .csv file if there is not one on folder. Then returns if as a df
    '''
    filename = 'telco.csv'
    if os.path.exists(filename) and use_cache:
        print('Reading from csv file...')
        return pd.read_csv(filename)
      
    query = '''
    SELECT * 
    FROM customers
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('telco_churn'))
    print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df