import pandas as pd
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.metrics import accuracy_score, roc_curve, auc, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, log_loss, brier_score_loss
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit
from sklearn.inspection import permutation_importance as pimp
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import sys


data_dir = '/mnt/c/Users/Shina/Documents/ml_qsar'
outdir_fig = '/mnt/c/Users/Shina/Documents/ml_qsar/output/figure'
outdir_mod = '/mnt/c/Users/Shina/Documents/ml_qsar/output/model'
outdir_csv = '/mnt/c/Users/Shina/Documents/ml_qsar/output/csv'
outdir = '/mnt/c/Users/Shina/Documents/ml_qsar/output'

# Read train test
X_train_rd = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
X_test_rd = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

# Replace label with integers
replace_dico = {'active':1, 'inactive':0}
X_train_rd['activity'] = X_train_rd['activity'].replace(replace_dico)
X_test_rd['activity'] = X_test_rd['activity'].replace(replace_dico)

#separate variables X and y
X_train = X_train_rd.iloc[:, :-3]
y_train = X_train_rd.iloc[:, -1]
X_test = X_test_rd.iloc[:, :-3]
y_test = X_test_rd.iloc[:, -1]

X_train_selected = X_train.copy()
X_test_selected = X_test.copy()

# Create print output
print_output = "MODELLING ECFP4E\n\n"

# based on result of hyperparamete tuning
best_params_rf = {'class_weight': 'balanced', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 10}
best_params_sv = {'C': 0.1, 'coef0': 0.0, 'degree': 1, 'gamma': 'scale', 'kernel': 'linear', 'shrinking': True}

# define classifiers
classifiers = [
    (RandomForestClassifier(random_state=42), "RF"),
    (RandomForestClassifier(random_state=42, **best_params_rf), "RF Tuned"),
    (SVC(probability=True, random_state=42), "SVM"),
    (SVC(probability=True, random_state=42, **best_params_sv), "SVM Tuned"),
    # (LogisticRegression(random_state=42), "LR")
]


sets = [
    (X_train_selected, y_train, "train"),
    (X_test_selected, y_test, "test"),
    # (X_ext_selected, y_ext, "external")
]

scores = defaultdict(list)
cross_val = defaultdict(list)

kappa_scorer = make_scorer(cohen_kappa_score)

scoring_cv = ['accuracy', 'balanced_accuracy', 'neg_log_loss', 'neg_brier_score', 'f1', 'f1_weighted', 'matthews_corrcoef', kappa_scorer]
metrics_tt = [(accuracy_score, "accuracy"), (bas, "balanced accuracy"), (f1_score, "f1"),
              (f1_score, "f1_weighted"), (cohen_kappa_score, "kappa"), (matthews_corrcoef, "mcc"),
              (log_loss, "log_loss"), (brier_score_loss, "brier_score"), (auc, "AUC")
]
metrics_tt_rand = [(accuracy_score, "random_accuracy"), (bas, "random_balanced_accuracy"), (f1_score, "random_f1"),
                   (f1_score, "random_f1_weighted"), (cohen_kappa_score, "random_kappa"), (matthews_corrcoef, "random_mcc"),
                   (log_loss, "random_logloss"), (brier_score_loss, "random_brier_score"), (auc, "random_AUC")
]

# Perform modelling on each classifier
for i, (clf, clf_name) in enumerate(classifiers):
    print_output += f"Classifier: {clf_name}\n\n"
    clf.fit(X_train_selected, y_train)
    scores["Classifiers"].append(clf_name)
    cross_val["Classifiers"].append(clf_name)
    for metric_cv in scoring_cv:
        train_cv = cross_validate(clf, X_train_selected, y_train, cv = 5, n_jobs=-1, return_train_score=True, scoring=metric_cv)
        print_output += f"Cross Validation ({clf_name} - {metric_cv}):\n \
                        training: {np.mean(train_cv['train_score']):.4f} ± {np.std(train_cv['train_score']):.4f} \
                        test: {np.mean(train_cv['test_score']):.4f} ± {np.std(train_cv['test_score']):.4f}\n\n"
        cross_val[f"{metric_cv}_train"].append(f"{np.mean(train_cv['train_score']):.4f} ± {np.std(train_cv['train_score']):.4f}")
        cross_val[f"{metric_cv}_test"].append(f"{np.mean(train_cv['test_score']):.4f} ± {np.std(train_cv['test_score']):.4f}")
        
    np.random.seed(42)
    fpr = defaultdict(list)
    for x_set, y_set, set_name in sets:
        y_pred = clf.predict(x_set)
        y_pred_proba = clf.predict_proba(x_set)[:, 1]
        y_randomized = np.random.permutation(y_set)
        y_pred_randomized = clf.predict(x_set)
        y_pred_randomized_proba = clf.predict_proba(x_set)[:, 1]
        
        for metric_def, metric_name in metrics_tt:
            if metric_name == "f1_weighted":
                metric = metric_def(y_set, y_pred, average='weighted')
                print_output += f"{metric_name} for {set_name} set: {metric:.4f}\n"
                score_name = f"{metric_name}_{set_name}"
                scores[score_name].append(metric)
            elif metric_name == "log_loss" or metric_name == "brier_score":
                metric = metric_def(y_set, y_pred_proba)
                print_output += f"{metric_name} for {set_name} set: {metric:.4f}\n"
                score_name = f"{metric_name}_{set_name}"
                scores[score_name].append(metric)
            elif metric_name == "AUC":
                fpr1, tpr1, thresholds1 = roc_curve(y_set, y_pred_proba)
                metric = metric_def(fpr1, tpr1)
                print_output += f"{metric_name} for {set_name} set: {metric:.4f}\n"
                score_name = f"{metric_name}_{set_name}"
                scores[score_name].append(metric)
            else:
                metric = metric_def(y_set, y_pred)
                print_output += f"{metric_name} for {set_name} set: {metric:.4f}\n"
                score_name = f"{metric_name}_{set_name}"
                scores[score_name].append(metric)

        for metric_def_rand, metric_name_rand in metrics_tt_rand:
            if metric_name_rand == "random_f1_weighted":
                metric_rand = metric_def_rand(y_randomized, y_pred_randomized, average='weighted')
                print_output += f"{metric_name_rand} for {set_name} set: {metric_rand:.4f}\n"
                score_name = f"{metric_name_rand}_{set_name}"
                scores[score_name].append(metric_rand)
            elif metric_name_rand == "random_log_loss" or metric_name_rand == "random_brier_score":
                metric_rand = metric_def_rand(y_randomized, y_pred_randomized_proba)
                print_output += f"{metric_name_rand} for {set_name} set: {metric_rand:.4f}\n"
                score_name = f"{metric_name_rand}_{set_name}"
                scores[score_name].append(metric_rand)
            elif metric_name_rand == "random_AUC":
                fpr1, tpr1, thresholds1 = roc_curve(y_randomized, y_pred_randomized_proba)
                metric_rand = metric_def_rand(fpr1, tpr1)
                print_output += f"{metric_name_rand} for {set_name} set: {metric_rand:.4f}\n"
                score_name = f"{metric_name_rand}_{set_name}"
                scores[score_name].append(metric_rand)
            else:
                metric_rand = metric_def_rand(y_randomized, y_pred_randomized)
                print_output += f"{metric_name_rand} for {set_name} set: {metric_rand:.4f}\n"
                score_name = f"{metric_name_rand}_{set_name}"
                scores[score_name].append(metric_rand)

        
        #confusion matrix
        cmap = plt.cm.Blues
        cm = confusion_matrix(y_set, y_pred)
        print_output += f"\n\n\nConfusion matrix for {set_name} set:\n{cm}\n\n"

        #plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
        disp.plot(cmap=cmap, colorbar=True, values_format=".4g")
        # Customize the colorbar
        cax = plt.gcf().axes[-1]
        cax.tick_params(colors='black')
        cax.yaxis.label.set_color('white')
        plt.title(f"Confusion Matrix ({set_name}) for\n{clf_name}")
        plt.savefig(f"{outdir_fig}/{clf_name.replace(' ', '_')}_CM_{set_name}.png", dpi=600, bbox_inches="tight")
        plt.clf()

        
        X_table = x_set.copy()
        X_table["activity"] = y_set
        X_table["prediction"] = y_pred
        X_table["prob_Active"] = y_pred_proba # probability of prediction
        # X_table.to_csv(f"{outdir_csv}/{clf_name.replace(' ', '_')}_{set_name}_data_table_ecfp4e_chembl.csv", index=True)
        X_table = X_table[["activity", "prediction", "prob_Active"]]
        X_table.to_csv(f"{outdir_csv}/{clf_name.replace(' ', '_')}_{set_name}_prediction.csv", index=True)

    
    
    #Save model with pickle
    with open(f"{outdir_mod}/{clf_name.replace(' ', '_')}.pkl", "wb") as file1:
        pickle.dump(clf, file1)
    
    
    # Plot average roc
    
    X_ffs = X_train_selected.to_numpy()
    y_ffs = y_train.to_numpy()
    
    cv = StratifiedKFold(n_splits=5)
    #classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X_ffs, y_ffs)):
        clf.fit(X_ffs[train], y_ffs[train])
        viz = RocCurveDisplay.from_estimator(
            clf,
            X_ffs[test],
            y_ffs[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Cross Validation mean ROC curve for {clf_name}\n(Positive class label = active)",
    )
    ax.axis("square")
    ax.legend(loc="lower right", fontsize="8")
    
    plt.savefig(f"{outdir_fig}/{clf_name.replace(' ', '_')}_ROC_average.png", dpi=600, bbox_inches="tight")
    plt.clf()

    score_df = pd.DataFrame(scores).set_index("Classifiers")

# Save the print output to a text file
with open(f"{outdir}/model_result_overview.txt", "w") as file:
    file.write(print_output)

# save metrics to csv
score_df_transpose = score_df.transpose()
score_df_transpose.to_csv(f"{outdir_csv}/metrics_table.csv", index=True)
cross_val_df = pd.DataFrame(cross_val).set_index("Classifiers")
cross_val_df_transpose = cross_val_df.transpose()
cross_val_df_transpose.to_csv(f"{outdir_fig}/CV_metrics_table.csv", index=True)

