# disaster-response-pipeline
Data Science Nanodegree Project 

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

## Project Motivation <a name="motivation"></a>
In this project I was interested in gaining a better understanding of the process of building ETL, NLP, and Machine Learning pipelines - with the actual model performance being secondary. This project does this by taking two CSV files containing information on messages sent during a disaster event. The relevant pipelines have been built for a machine learning model to then classify these messages.
## File Descriptions <a name="files"></a>
Within this repository there are 3 folders: data, models, and app.

The data folder contains 2 csv files, disaster_messages.csv and disaster_categories.csv, as well as a python script process_data.py. disaster_messages.csv contains the message id, the message itself, the original message before any translation or modification, and the genre of the message. disaster_categories.csv contains the message id and a column 'categories' describing the categories that that message id falls within. The script process_data.py takes the filepaths of these 2 datasets and the filepath for a database. It merges the messages and categories into a single dataframe and cleans the dataframe by splitting the categories column into separate, clearly named columns, then converting values to binary, and finally dropping any duplicates. This clean data is then stored in a SQLite database in the specified database file path.

The models folder contains a python script train_classifier.py. This script takes the file path for the database and a file path for the model. It then creates and trains a classifier - in this case a multi-output classifier using a random forest classifier - which is then stored as a pickle file in the specified model file path location.

The app folder contains a folder of html templates and a python script, run.py. This script runs a web app, which contains 2 visualisations created from the SQLite database, and shows how new messages would be classified by the model.
## Results <a name="results"></a>

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

