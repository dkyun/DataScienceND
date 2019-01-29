import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the data. Takes in the data as csv files
    :param messages_filepath: Takes the filepath to the messages data
    :param categories_filepath: Takes the filepath to the categories data
    :return: A merged dataframe of both datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    Cleans the dataset to be used in further modeling
    :param df: Takes in the merged dataframe
    :return: Returns the cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[1]
    category_colnames = row.apply(lambda i: i[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1::]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    df = df.drop(['categories'], axis=1)
    df= pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    # Clean the related column, as it has multiple output, which crashes classification report
    df.related.replace(2,0, inplace=True)
    return df

def save_data(df, database_filename):
    """
    Simply saves the data, nothing else
    :param df: Take a dataframe as input, this dataframe should be cleaned.
    :param database_filename: Takes in a filepath, this is the name and the location of the saved db.
    :return: Returns nothing, just saves the file
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponseData', engine, index=False, if_exists='replace')

def main():
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