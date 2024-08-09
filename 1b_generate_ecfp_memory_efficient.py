import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress verbose
RDLogger.DisableLog('rdApp.*')

data_dir = '/mnt/c/Users/Shina/Documents/chembl_set_new'
out_dir = '/mnt/c/Users/Shina/Documents/ml_qsar/fp_out'

# Read the SDF file
files = ["file_1.sdf", "file_2.sdf"]

for file in files:
    # Read the SDF file
    suppl = Chem.SDMolSupplier(f'{data_dir}/{file}.sdf')
    
    # Open the CSV file for writing
    csv_file_path = f'{out_dir}/{file}.csv'
    with open(csv_file_path, 'w') as csv_file:
        # Write the header to the CSV file
        csv_file.write("molecule," + ",".join(["ECFP_" + str(i + 1) for i in range(2048)]) + "\n")
    
        # Generate ECFP4 fingerprints for each molecule and write to the CSV file
        for mol in suppl:
            if mol is not None:
                # Get the compound ID from the molecule properties
                compound_id = mol.GetProp("chembl_id") if mol.HasProp("chembl_id") else ""
    
                # Generate ECFP4 fingerprint with radius 2 and 2048 bits
                ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
                # Convert the fingerprint to a bit string
                fingerprint_str = ecfp.ToBitString()
    
                # Write the fingerprint to the CSV file
                csv_file.write(compound_id + "," + ",".join(fingerprint_str) + "\n")
