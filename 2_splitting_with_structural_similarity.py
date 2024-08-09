# Import

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.model_selection import train_test_split
import sys


data_dir = '/mnt/c/Users/Shina/Documents/ml_qsar/input_data'
out_dir = '/mnt/c/Users/Shina/Documents/ml_qsar/results/preprocs_fp_efcp4'

# the print output records key output
print_output = "PREPROCESS AND SPLITTING for ECFP4 (Summary Outlook)\n\n"

#import data
df = pd.read_csv(f"{data_dir}/ECFP4_fp.csv", index_col="molecule")
df = df.dropna()
print_output += f"Shape of initial dataframe: {df.shape}\n"

#class threshold (file contains pIC50 column)
activity = []
for i in df['pIC50']:
  if float(i) >= 5.100000:
    activity.append("active")
  else:
    activity.append("inactive")
df['activity'] = activity
print_output += f"Shape of class label added: {df.shape}\n"
print_output += f"Shape of class label added: {df['activity'].value_counts()}\n"

#separate variables X and y
X = df.iloc[:, :-3] # range specific for my file
y = df.iloc[:, -1]
print_output += f"The shape of X is {X.shape} and that of y is {y.shape}\n"


# Split data by structural diversity and activity
def rdkit_split(dataset, train_size=0.8, random_state=0):
    # Count the number of active and inactive compounds
    active_count = dataset['activity'].value_counts()['active']
    inactive_count = dataset['activity'].value_counts()['inactive']
    #print_output += f"Active count is: {active_count} and Inactive count is: {inactive_count}\n"
    
    # DataFrames for active and inactive compounds
    active_df = dataset[X_dataset['activity'] == 'active']
    inactive_df = dataset[X_dataset['activity'] == 'inactive']
    
    # Calculate the desired number of active and inactive compounds for training and test sets
    active_train_count = int(active_count * train_size)
    inactive_train_count = int(inactive_count * train_size)
    
    active_test_count = active_count - active_train_count
    inactive_test_count = inactive_count - inactive_train_count
    
    active_train_count, active_test_count, inactive_train_count, inactive_test_count
    
    #print_output += f"active_train_count, active_test_count, inactive_train_count, inactive_test_count\n{active_train_count, active_test_count, inactive_train_count, inactive_test_count}\n"
    
    # Shuffle the active and inactive compounds
    active_df = active_df.sample(frac=1, random_state=random_state)
    inactive_df = inactive_df.sample(frac=1, random_state=random_state)
    
    # Convert SMILES strings to RDKit molecules
    active_df['mol'] = active_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    inactive_df['mol'] = inactive_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    
    # Calculate molecular fingerprints (ECFP4) for each compound
    active_df['fp'] = active_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    inactive_df['fp'] = inactive_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    
    # Create lists to store the selected compounds for the training and test sets
    train_compounds = []
    test_compounds = []
    
    # Select compounds for the training set
    active_train_size = int(active_count * train_size)
    inactive_train_size = int(inactive_count * train_size)
    
    train_compounds = active_df.sample(n=active_train_size, random_state=random_state).index.tolist() + \
                      inactive_df.sample(n=inactive_train_size, random_state=random_state).index.tolist()
    
    # Select compounds for the test set
    test_compounds = list(set(active_df.index.tolist() + inactive_df.index.tolist()) - set(train_compounds))


    # Training and test data frame
    train_df = dataset.loc[train_compounds]
    test_df = dataset.loc[test_compounds]
    
    # Convert the SMILES strings to RDKit molecules and calculate molecular fingerprints
    train_df['mol'] = train_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    test_df['mol'] = test_df['isomeric_smiles'].apply(Chem.MolFromSmiles)
    
    # Calculate molecular fingerprints (ECFP4) for each compound
    train_df['fp'] = train_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    test_df['fp'] = test_df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
        
    # Calculate fingerprints for compounds in the training and test set
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in train_df['mol']]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in test_df['mol']]

    # Calculate the average similarity score between the training and test sets
    #def calculate_similarity(fp1, fp2):
    #    return DataStructs.TanimotoSimilarity(fp1, fp2)
    similarity_scores = []
    for train_index, train_row in train_df.iterrows():
        train_fp = train_row['fp']
        similarities = []
    
        for test_index, test_row in test_df.iterrows():
            test_fp = test_row['fp']
            #similarity = calculate_similarity(train_fp, test_fp)
            similarity = DataStructs.TanimotoSimilarity(train_fp, test_fp)
            similarities.append(similarity)
    
        average_similarity = sum(similarities) / len(similarities)
        similarity_scores.append(average_similarity)
    
    average_similarity_train_test = sum(similarity_scores) / len(similarity_scores)

    # Calculate the average similarity score within the training set
    train_similarity_scores = []
    for i in range(len(train_fps)):
        for j in range(i + 1, len(train_fps)):
            similarity = DataStructs.FingerprintSimilarity(train_fps[i], train_fps[j])
            train_similarity_scores.append(similarity)
    
    average_similarity_train = sum(train_similarity_scores) / len(train_similarity_scores)
    
    # Calculate the average similarity score within the test set
    test_similarity_scores = []
    for i in range(len(test_fps)):
        for j in range(i + 1, len(test_fps)):
            similarity = DataStructs.FingerprintSimilarity(test_fps[i], test_fps[j])
            test_similarity_scores.append(similarity)
    
    average_similarity_test = sum(test_similarity_scores) / len(test_similarity_scores)
    #drop mol and fp columns
    train_df = train_df.drop(columns=['mol', 'fp'])
    test_df = test_df.drop(columns=['mol', 'fp'])
    # Return train and test compounds along with similarities
    return train_df, test_df, average_similarity_train_test, average_similarity_train, average_similarity_test

    
# Perform the splitting
X_train_rd, X_test_rd, X_train_test_sim, X_train_sim, X_test_sim = rdkit_split(X, train_size=0.8, random_state=7)
print_output += f"Similarity in dataset: {X_train_test_sim}\n"
print_output += f"Similarity in training set: {X_train_sim}\n"
# print_output += f"Similarity in test set: {X_test_sim}\n"
# print(X_train_test_sim, X_train_sim, X_test_sim)

# Save train test and external to csv
X_train_rd.to_csv(f"{out_dir}/train_fp.csv", index=True)
X_test_rd.to_csv(f"{out_dir}/test_fp.csv", index=True)
# X_external.to_csv(f"{out_dir}/external_fp.csv", index=True)

# Save the print output to a text file
with open(f"{out_dir}/print_output.txt", "w") as file:
    file.write(print_output)