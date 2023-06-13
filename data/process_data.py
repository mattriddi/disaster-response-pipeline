import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message and category data from their respective
    file locations and merge them into one dataframe
    
    INPUT: Takes two filepaths
    
    OUTPUT: Returns the files from the two filepaths as one merged dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how = 'left')
    return df

def clean_data(df):
    """
    This function renames the category columns in the dataframe, converts the 
    category data to a numeric format, and drops duplicates
    
    INPUT: The merged dataframe
    
    OUTPUT: A dataframe with the columns renamed, the category data converted
    to a numerical format, and duplicates dropped
    """
    
    categories = (df['categories'].str.split(';', expand=True))
    row = categories.iloc[0]
    category_colnames = row.str.partition('-')[0]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
        
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    df.drop(df.loc[df['related']==2].index, inplace=True)
    return df


def save_data(df, database_filename):
    """
    This function takes in a dataframe and a database filename and saves the 
    dataframe in a sql database of that filename    
    """
    engine = create_engine(('sqlite:///' + database_filename))
    df.to_sql('disaster_response_table', engine, index=False)
    pass  


def main():
    """
    This function loads, cleans and saves the data
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()