# Machine learning demo using diamonds dataset 

# Import packages
import pandas as pd
import pathlib as pl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

#%% Do some basic pre-processing on the dataset

# Define path to diamonds dataset
path_to_diamonds = r"F:\For_Carl\scripts\Machine_learning_demo\diamonds.csv"

# Turn path to diamonds dataset into a pathlib object
ds_path = pl.Path(path_to_diamonds)

# Import dataset using pandas
df = pd.read_csv(ds_path)

# Transform the "object" dtype columns into categories (int64)
for col in df.select_dtypes(include="object"):
    df[col] = df[col].replace({cut:label for cut, label in zip(df[col].unique(), range(len(df[col].unique())))})

# Split the dataset into training and test portions
training_set, test_set = train_test_split(df, test_size=0.2)

#%% Train the XGBoost classification model (WITHOUT HP TUNING)

# Initialise XGBoost
boost = xgb.XGBClassifier()

# Fit the model to the data, and try to predict the "Cut"
boost = boost.fit(training_set.drop(["cut"], axis=1), training_set["cut"])

# Predict the cut of the what the unlabelled tests set and print the accuracy
predicted_labels = boost.predict(test_set.drop(["cut"], axis=1))
print(metrics.accuracy_score(test_set["cut"], predicted_labels))

#%% Train the XGBoost classification model (WITH HP TUNING)

space = {'max_depth': hp.quniform("max_depth", 3, 100, 1),
        'gamma': hp.uniform ('gamma', 1, 9),
        'reg_alpha' : hp.quniform('reg_alpha', 0, 1000, 1),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 50}

accuracies = []

def objective(space):
    boost = xgb.XGBClassifier(
                    n_estimators = space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight = int(space['min_child_weight']),
                    colsample_bytree = int(space['colsample_bytree']))
    
    # Fit the model to the data, and try to predict the "Cut"
    boost = boost.fit(training_set.drop(["cut"], axis=1), training_set["cut"])
    
    predicted_labels = boost.predict(test_set.drop(["cut"], axis=1))
    

    accuracy = metrics.accuracy_score(test_set["cut"], predicted_labels)
    print (f"Accuracy: {accuracy}")
    accuracies.append(accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }


trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 250,
                        trials = trials)

#%% Optimisation using GridSearch
# Define the hyperparameter grid
param_grid = {
    'max_depth': list(range(5, 20)),
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.5, 0.7, 1]
}

# Create the XGBoost model object
boost = xgb.XGBClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(boost, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(training_set.drop(["cut"], axis=1), training_set["cut"])

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
