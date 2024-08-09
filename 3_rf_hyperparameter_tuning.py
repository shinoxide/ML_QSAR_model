import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, log_loss, brier_score_loss
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sys

data_dir = '/mnt/c/Users/Shina/Documents/ml_qsar/results/preprocs_ecfp4e_c2_ne'
out_dir = '/mnt/c/Users/Shina/Documents/ml_qsar/tunning/ecfp4e_c2_ne'


train_rd = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
test_rd = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

# Replace label with integers
replace_dico = {'active':1, 'inactive':0}
train_rd['activity'] = train_rd['activity'].replace(replace_dico)
test_rd['activity'] = test_rd['activity'].replace(replace_dico)


#separate X and y
X_train = train_rd.iloc[:, :-3]
y_train = train_rd.iloc[:, -1]
X_test = test_rd.iloc[:, :-3]
y_test = test_rd.iloc[:, -1]



# Define the hyperparameters to tune
param_grid={'class_weight': [None, 'balanced', 'balanced_subsample'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_leaf': [2, 3, 5, 7, 9],
            'min_samples_split': [3, 4, 6, 8, 10],
            'n_estimators': [30, 50, 100, 200]
            #'max_features': ['sqrt', 'log2']
}


# Define scoring
scoring = ['accuracy', 'neg_log_loss', 'neg_brier_score']


# Create the Random Forest classifier
rf = RandomForestClassifier(random_state=42)
#rf = CalibratedClassifierCV(RandomForestClassifier(random_state=42))

# Define the search strategy (Grid Search in this case)
grid_search = GridSearchCV(rf, param_grid=param_grid, scoring=scoring, n_jobs=-1,
                           refit='neg_log_loss', cv=None, verbose=0, pre_dispatch='2*n_jobs',
                           return_train_score=False)


# Perform hyperparameter tuning on the training set
grid_search.fit(X_train, y_train)


# best hyperparameters
best_params = grid_search.best_params_

print_out = f"Tunning rf tune \n\n"
print_out += f"param_grid: {grid_search}\n\n"
print_out += f"Best hyperparameters: {best_params}\n\n"

# Applying best hyperparameters
best_rf = RandomForestClassifier(random_state=42, **best_params)
#best_rf = CalibratedClassifierCV(RandomForestClassifier(random_state=42, **best_params))

# Train the model on the combined training and validation sets
best_rf.fit(X_train, y_train)

# Evaluate on the train set
y_train_pred = best_rf.predict(X_train)
y_train_pred_proba = best_rf.predict_proba(X_train)[:, 1]
train_accuracy = accuracy_score(y_train, y_train_pred)
train_bas = bas(y_train, y_train_pred)
train_logloss = log_loss(y_train, y_train_pred_proba)
train_brier_score = brier_score_loss(y_train, y_train_pred_proba)
train_f1 = f1_score(y_train, y_train_pred)

# Evaluate on the test set
y_test_pred = best_rf.predict(X_test)
y_test_pred_proba = best_rf.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_test_pred)
test_bas = bas(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_pred_proba)
test_brier_score = brier_score_loss(y_test, y_test_pred_proba)
test_f1 = f1_score(y_test, y_test_pred)

cv_metric = ['accuracy', 'balanced_accuracy', 'neg_log_loss', 'neg_brier_score', 'f1']
cv = 10
for metric in cv_metric:
    cross_val = cross_val_score(best_rf, X_train, y_train, cv = cv, scoring=metric)
    print_out += f"Cross Validation {metric}: {np.mean(cross_val):.4f} ± {np.std(cross_val):.4f}:\n"

print_out += f"Accuracy on train, test: {train_accuracy:.4f}, {test_accuracy:.4f}\n"
print_out += f"Balanced Accuracy on train, test: {train_bas:.4f}, {test_bas:.4f}\n"
print_out += f"Log loss on train, test: {train_logloss:.4f}, {test_logloss:.4f}\n"
print_out += f"Brier Score on train, test: {train_brier_score:.4f}, {test_brier_score:.4f}\n"
print_out += f"F1 Score on train, test and external set: {train_f1:.4f}, {test_f1:.4f}\n"

# Save the print output to a text file
with open(f"{out_dir}/RF_tuned_ecfp4e_c2_ne.txt", "w") as file:
    file.write(print_out)