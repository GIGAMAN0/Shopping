# Shopping Customer Purchase Prediction

This project implements an AI to predict whether online shopping customers will complete a purchase based on various features of their browsing behavior.

## Introduction

The goal of this project is to build a nearest-neighbor classifier to predict whether or not a user will make a purchase during an online shopping session. Given information about a user's browsing behavior, such as the number of pages visited, the duration of the session, and the type of pages visited, the classifier will predict whether or not the user will complete a purchase.

## Files

- `shopping.py`: Contains functions to load data from a CSV file, train a machine learning model, and evaluate the model's performance.
- `shopping.csv`: Dataset containing information about online shopping sessions and whether or not a purchase was made.

## Usage

To run the program and evaluate the model's performance, use the following command:
'''
python shopping.py shopping.csv
'''

Replace `shopping.csv` with the path to the dataset file.

## How It Works

The program loads data from the provided CSV file using the `load_data` function. It then splits the data into a training set and a testing set. The `train_model` function is called to train a nearest-neighbor classifier on the training data. Finally, the model is used to make predictions on the testing data set, and the performance of the model is evaluated using the `evaluate` function.

## Acknowledgments

This project is part of the AI course by CS50.
