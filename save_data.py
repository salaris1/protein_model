
import pickle 
import os
import pandas as pd

# Load the TSV file

datafolder = "/home/salaris/protein_model/data/"

pickle_file_path = datafolder + "cas_data_512_v1/" #--> where to save the files 
# Create the directory if it doesn't exist

if not os.path.exists(pickle_file_path):
    os.makedirs(pickle_file_path)


file_path = '/home/salaris/protein_model/data/all_data_20240629_09.csvtrain_test.csv'
data = pd.read_csv(file_path, sep='\t',nrows=5000)
# data = data.tail(100)
# Display the first few rows of the data
data.head()

#%%
import re



#%%
# Function to split sequences and PTM sites into chunks
def split_into_chunks(row):
    sequence = row['seq']
    labels = row['class']
    chunk_size = 512
    
    # Calculate the number of chunks
    num_chunks = (len(sequence) + chunk_size - 1) // chunk_size
    
    # Split sequences and PTM sites into chunks
    sequence_chunks = [sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    
    # Create new rows for each chunk
    rows = []
    for i in range(num_chunks):
        new_row = row.copy()
        new_row['seq'] = sequence_chunks[i]
        new_row['class'] = labels
        rows.append(new_row)
    
    return rows

# Create a new DataFrame to store the chunks
chunks_data = []

# Iterate through each row of the original DataFrame and split into chunks
for _, row in data.iterrows():
    chunks_data.extend(split_into_chunks(row))

# Convert the list of chunks into a DataFrame
chunks_df = pd.DataFrame(chunks_data)

# Reset the index of the DataFrame
chunks_df.reset_index(drop=True, inplace=True)

# Display the first few rows of the new DataFrame
chunks_df.head()

#%%

from tqdm import tqdm
import numpy as np

# use sklaern train_test_split to split the data and stratify by labels 
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(chunks_df, test_size=0.2, stratify=chunks_df['class'], random_state=42)

#%%

import pandas as pd

# Assuming train_df and test_df are your dataframes
fraction = .100  # 100.0%

# Randomly select 100% of the data
reduced_train_df = train_df.sample(frac=fraction, random_state=42)
reduced_test_df = test_df.sample(frac=fraction, random_state=42)

#%%


# Extract sequences and PTM site labels from the reduced train and test DataFrames
train_sequences_reduced = reduced_train_df['seq'].tolist()
train_labels_reduced = reduced_train_df['class'].tolist()
test_sequences_reduced = reduced_test_df['seq'].tolist()
test_labels_reduced = reduced_test_df['class'].tolist()

# Save the lists to the specified pickle files


with open(pickle_file_path + "train_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_sequences_reduced, f)


with open(pickle_file_path + "test_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_sequences_reduced, f)

with open(pickle_file_path + "train_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_labels_reduced, f)

with open(pickle_file_path + "test_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_labels_reduced, f)