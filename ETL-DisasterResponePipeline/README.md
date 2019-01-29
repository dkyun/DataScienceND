# Disaster Response Pipeline Project
### Description:
This project was part of the Udacity Data Scientist Nanodegree.
The goal was to create an ETL Pipeline to classify messages to help Disaster Response Organizations. 
Below one can find the requirements for this project, as well as Instructions how to use the files on hand.
### Requirements:

* Flask==1.0.2
* plotly==3.5.0
* nltk==3.3
* scikit-learn==0.19.2
* SQLAlchemy==1.2.11
* pandas==0.23.4

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/mlmodel.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
