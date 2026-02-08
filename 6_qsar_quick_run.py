import pandas as pd
import pickle
import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit import DataStructs
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, roc_curve, auc, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, log_loss, brier_score_loss
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn_genetic import GAFeatureSelectionCV as gafs
import sys
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit
from sklearn.inspection import permutation_importance as pimp

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


# Generate ECFP for compound library

RDLogger.DisableLog('rdApp.*')

data_dir = 'data'
out_dir = 'mod_out'
os.makedirs(out_dir, exist_ok=True)
suppl = Chem.SDMolSupplier(f"{data_dir}/TableS1_Compound_Library.sdf")
data = []

for mol in suppl:
    if mol is not None:
        compound_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_str = ecfp.ToBitString()
        fingerprint_bits = list(fingerprint_str)
        fingerprint_bits.insert(0, compound_id)
        data.append(dict(zip(["ECFP_" + str(i) for i in range(2049)], fingerprint_bits)))

df = pd.DataFrame(data)
df.set_index('ECFP_0', inplace=True)
df.index.names = ['Compound ID']
df = df.reset_index()
df2 = pd.read_csv(f"{data_dir}/TableS1_Compound_Library.csv", encoding='ISO-8859-1')
df2 = df2.rename(columns={'Isomeric SMILES': 'isomeric_smiles'})

df = pd.merge(df, df2[['Compound ID', 'isomeric_smiles', 'pIC50']], on='Compound ID', how='left')

df.rename(columns={'Compound ID': 'molecule'}, inplace=True)
df.set_index('molecule', inplace=True)

df.to_csv(f"{data_dir}/ECFP4E.csv")
# df.head(20)



# Generate drugbank fingerprint

suppl = Chem.SDMolSupplier(f"{data_dir}/lig_drugbank.sdf")

data = []
for mol in suppl:
    if mol is not None:
        compound_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint_str = ecfp.ToBitString()
        fingerprint_bits = list(fingerprint_str)
        fingerprint_bits.insert(0, compound_id)
        data.append(dict(zip(["ECFP_" + str(i) for i in range(2049)], fingerprint_bits)))

df = pd.DataFrame(data)
df.set_index('ECFP_0', inplace=True)
df.index.names = ['molecule']
df.to_csv(f"{data_dir}/ECFP4E_drugbank_fp.csv")
# df.head(20)




# Splitting

df = pd.read_csv(f"{data_dir}/ECFP4E.csv", index_col="molecule")
df = df.dropna()

activity = []
for i in df['pIC50']:
  if float(i) >= 5.6900000:
    activity.append("active")
  else:
    activity.append("inactive")
df['activity'] = activity

#separate variables X and y
X = df.iloc[:, :-3]
y = df.iloc[:, -1]

X_data, X_ext, y_data, y_ext = train_test_split(X, y,
                                                    stratify=y,test_size=0.1,
                                                    random_state=42)

#lookup
map_isomeric_smiles = df["isomeric_smiles"].to_dict()
map_pIC50 = df["pIC50"].to_dict()
map_activity = df["activity"].to_dict()
X_dataset = X_data.copy()
X_dataset["isomeric_smiles"] = X_dataset.index.map(map_isomeric_smiles)
X_dataset["pIC50"] = X_dataset.index.map(map_pIC50)
X_dataset["activity"] = X_dataset.index.map(map_activity)
X_external = X_ext.copy()
X_external["isomeric_smiles"] = X_external.index.map(map_isomeric_smiles)
X_external["pIC50"] = X_external.index.map(map_pIC50)
X_external["activity"] = X_external.index.map(map_activity)

def rdkit_split(dataset, train_size=0.8):
    active_count = dataset['activity'].value_counts()['active']
    inactive_count = dataset['activity'].value_counts()['inactive']

    active_df = dataset[X_dataset['activity'] == 'active']
    inactive_df = dataset[X_dataset['activity'] == 'inactive']
    
    active_train_count = int(active_count * train_size)
    inactive_train_count = int(inactive_count * train_size)
    
    active_test_count = active_count - active_train_count
    inactive_test_count = inactive_count - inactive_train_count
    
    # active_train_count, active_test_count, inactive_train_count, inactive_test_count
    
    active_df = active_df.sample(frac=1, random_state=7)
    inactive_df = inactive_df.sample(frac=1, random_state=7)
    
    active_df['mol'] = active_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    inactive_df['mol'] = inactive_df['isomeric_smiles'].apply(Chem.MolFromSmiles)

    active_df['fp'] = active_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    inactive_df['fp'] = inactive_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    
    # Create lists for training and test sets
    train_compounds = []
    test_compounds = []
    
    active_train_size = int(active_count * train_size)
    inactive_train_size = int(inactive_count * train_size)
    
    train_compounds = active_df.sample(n=active_train_size, random_state=1).index.tolist() + \
                      inactive_df.sample(n=inactive_train_size, random_state=1).index.tolist()

    test_compounds = list(set(active_df.index.tolist() + inactive_df.index.tolist()) - set(train_compounds))

    train_df = dataset.loc[train_compounds]
    test_df = dataset.loc[test_compounds]

    train_df['mol'] = train_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    test_df['mol'] = test_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
 
    train_df['fp'] = train_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    test_df['fp'] = test_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in train_df['mol']]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in test_df['mol']]

    similarity_scores = []
    for train_index, train_row in train_df.iterrows():
        train_fp = train_row['fp']
        similarities = []
    
        for test_index, test_row in test_df.iterrows():
            test_fp = test_row['fp']
            similarity = DataStructs.TanimotoSimilarity(train_fp, test_fp)
            similarities.append(similarity)
    
        average_similarity = sum(similarities) / len(similarities)
        similarity_scores.append(average_similarity)
    
    average_similarity_train_test = sum(similarity_scores) / len(similarity_scores)

    train_similarity_scores = []
    for i in range(len(train_fps)):
        for j in range(i + 1, len(train_fps)):
            similarity = DataStructs.FingerprintSimilarity(train_fps[i], train_fps[j])
            train_similarity_scores.append(similarity)
    
    average_similarity_train = sum(train_similarity_scores) / len(train_similarity_scores)

    test_similarity_scores = []
    for i in range(len(test_fps)):
        for j in range(i + 1, len(test_fps)):
            similarity = DataStructs.FingerprintSimilarity(test_fps[i], test_fps[j])
            test_similarity_scores.append(similarity)
    
    average_similarity_test = sum(test_similarity_scores) / len(test_similarity_scores)
    train_df = train_df.drop(columns=['mol', 'fp'])
    test_df = test_df.drop(columns=['mol', 'fp'])

    return train_df, test_df, average_similarity_train_test, average_similarity_train, average_similarity_test

# Perform the splitting
X_train_rd, X_test_rd, X_train_test_sim, X_train_sim, X_test_sim = rdkit_split(X_dataset, train_size=0.89)
X_test_combined = pd.concat([X_test_rd, X_external], axis=0, ignore_index=False)

X_train_rd.to_csv(f"{data_dir}/train_fp.csv", index=True)
X_test_combined.to_csv(f"{data_dir}/test_fp.csv", index=True)

X_train = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
X_test = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

def similarity_test(trainfp, testfp):
    trainfp['mol'] = trainfp['isomeric_smiles'].apply(Chem.MolFromSmiles)
    testfp['mol'] = testfp['isomeric_smiles'].apply(Chem.MolFromSmiles)
    trainfp['fp'] = trainfp['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    testfp['fp'] = testfp['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))

    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in trainfp['mol']]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in testfp['mol']]

    similarity_scores = []
    for train_index, train_row in trainfp.iterrows():
        train_fp = train_row['fp']
        similarities = []
    
        for test_index, test_row in testfp.iterrows():
            test_fp = test_row['fp']
            similarity = DataStructs.TanimotoSimilarity(train_fp, test_fp)
            similarities.append(similarity)
    
        average_similarity = sum(similarities) / len(similarities)
        similarity_scores.append(average_similarity)
    
    average_similarity_train_test = sum(similarity_scores) / len(similarity_scores)
    
    # similarity in the training set
    train_similarity_scores = []
    for i in range(len(train_fps)):
        for j in range(i + 1, len(train_fps)):
            similarity = DataStructs.FingerprintSimilarity(train_fps[i], train_fps[j])
            train_similarity_scores.append(similarity)
    
    average_similarity_train = sum(train_similarity_scores) / len(train_similarity_scores)
    
    # similarity in the test set
    test_similarity_scores = []
    for i in range(len(test_fps)):
        for j in range(i + 1, len(test_fps)):
            similarity = DataStructs.FingerprintSimilarity(test_fps[i], test_fps[j])
            test_similarity_scores.append(similarity)
    
    average_similarity_test = sum(test_similarity_scores) / len(test_similarity_scores)
    return average_similarity_train_test, average_similarity_train, average_similarity_test

train_test_sim, train_sim, test_sim = similarity_test(X_train, X_test)
print(f"average similarity on whole data: {train_test_sim:.4f}")
print(f"average similarity on training data: {train_sim:.4f}")
print(f"average similarity on test data: {test_sim:.4f}")


# Hyperparameter tuning

X_train_rd = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
X_test_rd = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

replace_dico = {'active':1, 'inactive':0}
X_train_rd['activity'] = X_train_rd['activity'].replace(replace_dico)
X_test_rd['activity'] = X_test_rd['activity'].replace(replace_dico)

X_train = X_train_rd.iloc[:, :-3]
y_train = X_train_rd.iloc[:, -1]
X_test = X_test_rd.iloc[:, :-3]
y_test = X_test_rd.iloc[:, -1]

X_train_selected = X_train.copy()
X_test_selected = X_test.copy()




# hyperparameters to tune

param_grid={'class_weight': [None, 'balanced', 'balanced_subsample'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_leaf': [2, 3, 5, 7, 9],
            'min_samples_split': [3, 4, 6, 8, 10],
            'n_estimators': [30, 50, 100, 200]
}

scoring = ['accuracy', 'neg_log_loss', 'neg_brier_score']

# Random Forest classifier
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid=param_grid, scoring=scoring, n_jobs=-1,
                           refit='neg_log_loss', cv=None, verbose=0, pre_dispatch='2*n_jobs',
                           return_train_score=True)

grid_search.fit(X_train_selected, y_train)

best_params = grid_search.best_params_
print(f"best parameters: {best_params}")



# run modelling

X_train_rd = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
X_test_rd = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

replace_dico = {'active':1, 'inactive':0}
X_train_rd['activity'] = X_train_rd['activity'].replace(replace_dico)
X_test_rd['activity'] = X_test_rd['activity'].replace(replace_dico)

X_train = X_train_rd.iloc[:, :-3]
y_train = X_train_rd.iloc[:, -1]
X_test = X_test_rd.iloc[:, :-3]
y_test = X_test_rd.iloc[:, -1]

X_train_selected = X_train.copy()
X_test_selected = X_test.copy()

# Create print output
print_output = "MODELLING ECFP4E - C2 tuned (no ext)\n\n"

classifiers = [
    (RandomForestClassifier(random_state=42), "RF"),
    (RandomForestClassifier(random_state=42, **best_params), "RF Tuned"),
    # (SVC(probability=True, random_state=42), "SVM"),
    # (SVC(probability=True, random_state=42, **best_params_sv), "SVM Tuned"),
    # (LogisticRegression(random_state=42), "LR")
]


sets = [
    (X_train_selected, y_train, "train"),
    (X_test_selected, y_test, "test"),
]

scores = defaultdict(list)
cross_val = defaultdict(list)

kappa_scorer = make_scorer(cohen_kappa_score)

scoring_cv = ['accuracy', 'balanced_accuracy', 'neg_log_loss', 'neg_brier_score', 'f1', 'f1_weighted', 'matthews_corrcoef', kappa_scorer]

for i, (clf, clf_name) in enumerate(classifiers):
    print_output += f"Classifier: {clf_name}\n\n"
    clf.fit(X_train_selected, y_train)
    scores["Classifiers"].append(clf_name)
    cross_val["Classifiers"].append(clf_name)
    for metric_cv in scoring_cv:
        train_cv = cross_validate(clf, X_train_selected, y_train, cv = 10, n_jobs=-1, return_train_score=True, scoring=metric_cv)
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
        accuracy = accuracy_score(y_set, y_pred)
        balanced_acc = bas(y_set, y_pred)
        f1 = f1_score(y_set, y_pred)
        f1_weighted = f1_score(y_set, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_set, y_pred)
        mcc = matthews_corrcoef(y_set, y_pred)
        logloss = log_loss(y_set, y_pred_proba)
        brier_score = brier_score_loss(y_set, y_pred_proba)
        y_randomized = np.random.permutation(y_set)
        y_pred_randomized = clf.predict(x_set)
        y_pred_randomized_proba = clf.predict_proba(x_set)[:, 1]
        random_accuracy = accuracy_score(y_randomized, y_pred_randomized)
        random_bas = bas(y_randomized, y_pred_randomized)
        random_f1 = f1_score(y_randomized, y_pred_randomized)
        random_f1_weighted = f1_score(y_randomized, y_pred_randomized, average='weighted')
        random_kappa = cohen_kappa_score(y_randomized, y_pred_randomized)
        random_kappa = cohen_kappa_score(y_randomized, y_pred_randomized)
        random_ll = log_loss(y_randomized, y_pred_randomized_proba)
        random_bs = brier_score_loss(y_randomized, y_pred_randomized_proba)

        fpr1, tpr1, thresholds1 = roc_curve(y_set, y_pred_proba)
        roc_auc1 = auc(fpr1, tpr1)
        
        #Y_randomization
        fpr4, tpr4, thresholds4 = roc_curve(y_randomized, y_pred_randomized_proba)
        roc_auc4 = auc(fpr4, tpr4)

        print_output += f"Accuracy for {set_name} set: {accuracy:.4f}\n"
        print_output += f"Balanced Accuracy for {set_name} set: {balanced_acc:.4f}\n"
        print_output += f"AUC for {set_name}: {roc_auc1:.4f}\n"
        print_output += f"f1 score for {set_name} set: {f1:.4f}\n"
        print_output += f"Cohen's kappa score for {set_name} set: {kappa:.4f}\n"
        print_output += f"Matthews correlation coefficient for {set_name} set: {mcc:.4f}\n"
        print_output += f"Log loss for {set_name} set: {logloss:.4f}\n"
        print_output += f"Brier score for {set_name} set: {brier_score:.4f}\n"
        print_output += f"Accuracy for {set_name} set Y-Randomization: {random_accuracy:.4f}\n"
        print_output += f"Balanced Accuracy for {set_name} set Y-Randomization: {random_bas:.4f}\n"
        print_output += f"f1 score for {set_name} set Y-Randomization: {random_f1:.4f}\n"
        print_output += f"Log loss for {set_name} set Y-Randomization: {random_ll:.4f}\n"
        print_output += f"Brier score for {set_name} set Y-Randomization: {random_bs:.4f}\n"
        print_output += f"AUC for y_random_{set_name}: {roc_auc4:.4f}\n\n"
        
        #confusion matrix
        cm = confusion_matrix(y_set, y_pred)
        print_output += f"Confusion matrix for {set_name} set:\n{cm}\n\n\n"

        #plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
        disp.plot()
        plt.title(f"Confusion Matrix ({set_name}) for\n{clf_name}")
        plt.savefig(f"{out_dir}/{clf_name.replace(' ', '_')}_CM_{set_name}_ecfp4e_c2_ne.png", dpi=600, bbox_inches="tight")
        plt.clf()

        
        X_table = x_set.copy()
        X_table["activity"] = y_set
        X_table["prediction"] = y_pred
        X_table["prob_Active"] = y_pred_proba
        X_table.to_csv(f"{out_dir}/{clf_name.replace(' ', '_')}_{set_name}_data_table_ecfp4e_c2_ne.csv", index=True)
        X_table = X_table[["activity", "prediction", "prob_Active"]]
        X_table.to_csv(f"{out_dir}/{clf_name.replace(' ', '_')}_{set_name}_proba_ecfp4e_c2_ne.csv", index=True)

        for metric, metric_name in [(accuracy, "accuracy"), (balanced_acc, "balanced accuracy"), (f1, "f1"),
                                    (kappa, "kappa"), (mcc, "mcc"), (logloss, "logloss"), (brier_score, "brier_score"),
                                    (roc_auc1, "AUC"), (random_accuracy, "random_accuracy"), (random_f1, "random_f1"),
                                    (random_ll, "random_logloss"), (random_bs, "random_brier_score")
                                   ]:
            score_name = f"{metric_name}_{set_name}"
            scores[score_name].append(metric)

    score_df = pd.DataFrame(scores).set_index("Classifiers")
    
    #Save model with pickle
    with open(f"{out_dir}/{clf_name.replace(' ', '_')}_ecfp4e_c2_ne.pkl", "wb") as file1:
        pickle.dump(clf, file1)
    
    #plot average roc
    X_ffs = X_train_selected.to_numpy()
    y_ffs = y_train.to_numpy()
    
    cv = StratifiedKFold(n_splits=10)
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
        title=f"Cross Validation mean ROC curve for {clf_name} (ECFP4E - C2)\n(Positive class label = active)",
    )
    ax.axis("square")
    ax.legend(loc="lower right", fontsize="6")
    
    plt.savefig(f"{out_dir}/{clf_name.replace(' ', '_')}_ROC_average_ecfp4e_c2_ne.png", dpi=600, bbox_inches="tight")
    plt.clf()


with open(f"{out_dir}/model_output_ecfp4e_c2_ne.txt", "w") as file:
    file.write(print_output)
score_df_transpose = score_df.transpose()
score_df_transpose.to_csv(f"{out_dir}/metrics_table_ecfp4e_c2_ne.csv", index=True)
cross_val_df = pd.DataFrame(cross_val).set_index("Classifiers")
cross_val_df_transpose = cross_val_df.transpose()
cross_val_df_transpose.to_csv(f"{out_dir}/CV_metrics_table_ecfp4e_c2_ne.csv", index=True)




# Applicability Domain

X_train_rd = pd.read_csv(f"{data_dir}/train_fp.csv", index_col="molecule")
X_test_rd = pd.read_csv(f"{data_dir}/test_fp.csv", index_col="molecule")

X_drugbank = pd.read_csv(f"{data_dir}/ECFP4E_drugbank_fp.csv", index_col="molecule")

replace_dico = {'active':1, 'inactive':0}
X_train_rd['activity'] = X_train_rd['activity'].replace(replace_dico)
X_test_rd['activity'] = X_test_rd['activity'].replace(replace_dico)

X_train = X_train_rd.iloc[:, :-3]
y_train = X_train_rd.iloc[:, -1]
X_test = X_test_rd.iloc[:, :-3]
y_test = X_test_rd.iloc[:, -1]

X_train_selected = X_train.copy()
X_test_selected = X_test.copy()
X_ext_selected = X_ext.copy()

cols_list = X_train.columns.tolist()

X_drugbank_selected = X_drugbank.loc[:, cols_list]
X_drugbank_selected = X_drugbank_selected.dropna()

X_train = X_train_selected.to_numpy()
X_test = X_test_selected.to_numpy()
X_drugbank = X_drugbank_selected.to_numpy()


# Calculate Euclidean distances
def calculate_apd(training_set, test_compounds, z=0.5):
    distances = distance.cdist(training_set, training_set, 'euclidean')
    avg_distance = np.mean(distances[np.triu_indices(len(training_set), k=1)])
    lower_distances = distances[np.where(distances < avg_distance)]
    d = np.mean(lower_distances)
    sigma = np.std(lower_distances)
    apd = d + (z * sigma)
    
    nearest_neighbor_distances = []
    for test_compound in test_compounds:
        nearest_neighbor_distance = calculate_nearest_neighbor_distance(training_set, test_compound)
        nearest_neighbor_distances.append(nearest_neighbor_distance)
    return apd, nearest_neighbor_distances

def calculate_nearest_neighbor_distance(training_set, test_compound):
    distances = distance.cdist(training_set, [test_compound], 'euclidean')
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_distance = distances[nearest_neighbor_index][0]
    return nearest_neighbor_distance

threshold, dist_train = calculate_apd(X_train, X_train)
threshold_test, dist_test = calculate_apd(X_train, X_test)
threshold_drugbank, dist_drugbank = calculate_apd(X_train, X_drugbank)

print(f"threshold is: {threshold}")

X_combined = np.concatenate((X_train, X_test, X_ext), axis=0)

# pca_combined = PCA(n_components=1, random_state=42)
# pca_values_combined = pca_combined.fit_transform(X_combined)

# pca_values_train = pca_values_combined[:X_train.shape[0]]
# pca_values_test = pca_values_combined[X_train.shape[0]:X_train.shape[0]+X_test.shape[0]]

# pca = PCA(n_components=1, random_state=42)
# pca_values_drugbank = pca.fit_transform(X_drugbank)

within_domain_train = dist_train < threshold
within_domain_test = dist_test < threshold
within_domain_drugbank = dist_drugbank < threshold

df_train = pd.DataFrame(X_train, columns=X_train_selected.columns, index=X_train_selected.index)
# df_train['PCA'] = pca_values_train
df_train['distances'] = dist_train

df_test = pd.DataFrame(X_test, columns=X_test_selected.columns, index=X_test_selected.index)
# df_test['PCA'] = pca_values_test
df_test['distances'] = dist_test

df_drugbank = pd.DataFrame(X_drugbank, columns=X_drugbank_selected.columns, index=X_drugbank_selected.index)
df_drugbank['distances'] = dist_drugbank

print(f"train shape before AD:{df_train.shape}, test shape before AD:{df_test.shape}")
print(f"drugbank shape before AD: {df_drugbank.shape}")

df_within_domain_train = df_train.loc[(df_train['distances'] < threshold)]
df_within_domain_test = df_test.loc[(df_test['distances'] < threshold)]
df_within_domain_drugbank = df_drugbank.loc[(df_drugbank['distances'] < threshold)]

print(f"train shape after AD: {df_within_domain_train.shape}, test shape after AD: {df_within_domain_test.shape}")
print(f"drugbank shape after AD: {df_within_domain_drugbank.shape}")

df_within_domain_drugbank.to_csv(f"{out_dir}/drugbank_within_AD_ecfp4e_c2.csv", index=True)




# Screening

vs_data = ["drugbank"]

classifiers = ["RF", "RF Tuned"]

for clf_name in classifiers:
    with open(f"{out_dir}/{clf_name.replace(' ', '_')}_ecfp4e_c2_ne.pkl", "rb") as file1:
        clf = pickle.load(file1)
    for data in vs_data:
        X_df = pd.read_csv(f"{out_dir}/{data}_within_AD_ecfp4e_c2.csv", index_col="molecule")
        X_df = X_df.dropna()
        X_df = X_df.iloc[:, :-1]
        X_data = X_df.copy()
        vs_pred = clf.predict(X_data)
        vs_pred_prob = clf.predict_proba(X_data)[:,1]
        vs_pred_data = X_data.copy()
        vs_pred_data['Predicted_activity'] = vs_pred
        vs_pred_data['Prob_active'] = vs_pred_prob
        vs_pred_exp = vs_pred_data[['Predicted_activity', 'Prob_active']]
        vs_pred_exp = vs_pred_exp.sort_values('Prob_active', ascending=False)
        vs_pred_exp.to_csv(f"{out_dir}/vs_{data}_{clf_name.replace(' ', '_')}_ecfp4e_c2_ne.csv", index=True)
        print(f"vs_{data}_{clf_name.replace(' ', '_')}_ecfp4e_c2 written to csv")
print(f"\nALL DONE")