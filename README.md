# Template for a process to be deployed in the data cage

This is a sample project with a representative set of resources to deploy a data processing machine learning algorithm in a Datavillage cage.
The example is using XGBoost library and its classification example: https://xgboost.readthedocs.io/en/stable/get_started.html

__TL;DR__ : clone this repo and edit the `process.py` file to adapt it for your own use casse


## Template Use case


To make this template closer to a real use case,
it implements 3 event processing logic
 1. TRAIN event to train a classification model on a data batch
 2. PREDICT event to predict classification classes on a data batch using a previously trained event
 3. INFER event to infer (predict) a classification class on one single case at a time

All these steps are defined in distinct functions of the process.py file.
Simply adapt this function when addressing your own use case


## Content of this repo

Have a look at the main template repository to get familiar with the template structure
https://github.com/datavillage-me/cage-process-template

## Infering event

For testing the inference, you can send an event with that content:
```
{
   "type": "INFER",
   "data": {
      "sepal length (cm)": 5.9,
      "sepal width (cm)": 3.0,
      "petal length (cm)": 5.1,
      "petal width (cm)": 1.8
   }
}
```
You should see in the log of the algorithm that it is categorize as a "virginica" Iris species
