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
suppl = Chem.SDMolSupplier(f"{data_dir}/d1c3_optimised_97.sdf")
#suppl = Chem.SDMolSupplier('/mnt/c/Users/Shina/Documents/Schrodinger/vs_en/ligprep_vs_en_3-out.sdf')

# Create an empty list to store the dictionaries
data = []

# Generate ECFP4 fingerprints for each molecule and store in the list
for mol in suppl:
    if mol is not None:
        # Get the compound ID from the molecule properties
        compound_id = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

        # Generate ECFP4/ECFP6 fingerprint with radius 2/3 and 2048 bits
        # ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048, useChirality=True, useFeatures=False)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        # Convert the fingerprint to a bit string
        fingerprint_str = ecfp.ToBitString()
        fingerprint_bits = list(fingerprint_str)

        # Compound ID as a column
        fingerprint_bits.insert(0, compound_id)

        # Append the fingerprint bits as a dictionary to the list
        #data.append(dict(zip(range(1025), fingerprint_bits)))
        data.append(dict(zip(["ECFP_" + str(i) for i in range(2049)], fingerprint_bits)))

# DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Set the compound ID as the DataFrame index
df.set_index('ECFP_0', inplace=True)
df.index.names = ['molecule']

# Save the DataFrame to a CSV file
df.to_csv(f"{out_dir}/ECFP4E_chembl.csv")
# df.head(20)