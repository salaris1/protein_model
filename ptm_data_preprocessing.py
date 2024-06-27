import pandas as pd

# Load the TSV file
file_path = '/home/salaris/fine_tuning/protein_data.tsv'
data = pd.read_csv(file_path, sep='\t')

# Display the first few rows of the data
data.head()

import re

def get_ptm_sites(row):
    # Extract the positions of modified residues from the 'Modified residue' column
    modified_positions = [int(i) for i in re.findall(r'MOD_RES (\d+)', row['Modified residue'])]
    
    # Create a list of zeros of length equal to the protein sequence
    ptm_sites = [0] * len(row['Sequence'])
    
    # Replace the zeros with ones at the positions of modified residues
    for position in modified_positions:
        # Subtracting 1 because positions are 1-indexed, but lists are 0-indexed
        ptm_sites[position - 1] = 1
    
    return ptm_sites

# Apply the function to each row in the DataFrame
data['PTM sites'] = data.apply(get_ptm_sites, axis=1)

# Display the first few rows of the updated DataFrame
data.head()


# Function to split sequences and PTM sites into chunks
def split_into_chunks(row):
    sequence = row['Sequence']
    ptm_sites = row['PTM sites']
    chunk_size = 1000
    
    # Calculate the number of chunks
    num_chunks = (len(sequence) + chunk_size - 1) // chunk_size
    
    # Split sequences and PTM sites into chunks
    sequence_chunks = [sequence[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    ptm_sites_chunks = [ptm_sites[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    
    # Create new rows for each chunk
    rows = []
    for i in range(num_chunks):
        new_row = row.copy()
        new_row['Sequence'] = sequence_chunks[i]
        new_row['PTM sites'] = ptm_sites_chunks[i]
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



from tqdm import tqdm
import numpy as np

# Function to split data into train and test based on families
def split_data(df):
    # Get a unique list of protein families
    unique_families = df['Protein families'].unique().tolist()
    np.random.shuffle(unique_families)  # Shuffle the list to randomize the order of families
    
    test_data = []
    test_families = []
    total_entries = len(df)
    total_families = len(unique_families)
    
    # Set up tqdm progress bar
    with tqdm(total=total_families) as pbar:
        for family in unique_families:
            # Separate out all proteins in the current family into the test data
            family_data = df[df['Protein families'] == family]
            test_data.append(family_data)
            
            # Update the list of test families
            test_families.append(family)
            
            # Remove the current family data from the original DataFrame
            df = df[df['Protein families'] != family]
            
            # Calculate the percentage of test data and the percentage of families in the test data
            percent_test_data = sum(len(data) for data in test_data) / total_entries * 100
            percent_test_families = len(test_families) / total_families * 100
            
            # Update tqdm progress bar with readout of percentages
            pbar.set_description(f'% Test Data: {percent_test_data:.2f}% | % Test Families: {percent_test_families:.2f}%')
            pbar.update(1)
            
            # Check if the 20% threshold for test data is crossed
            if percent_test_data >= 20:
                break
    
    # Concatenate the list of test data DataFrames into a single DataFrame
    test_df = pd.concat(test_data, ignore_index=True)
    
    return df, test_df  # Return the remaining data and the test data

# Split the data into train and test based on families
train_df, test_df = split_data(chunks_df)




import pandas as pd

# Assuming train_df and test_df are your dataframes
fraction = 0.105  # 10.5%

# Randomly select 10.5% of the data
reduced_train_df = train_df.sample(frac=fraction, random_state=42)
reduced_test_df = test_df.sample(frac=fraction, random_state=42)

# Split the reduced dataframes into sequences and PTM sites
#train_sequences = reduced_train_df['Sequence']
#train_ptm_sites = reduced_train_df['PTM sites']
#test_sequences = reduced_test_df['Sequence']
#test_ptm_sites = reduced_test_df['PTM sites']

# Save the reduced data as pickle files
#train_sequences.to_pickle('train_sequences.pkl')
#train_ptm_sites.to_pickle('train_ptm_sites.pkl')
#test_sequences.to_pickle('test_sequences.pkl')
#test_ptm_sites.to_pickle('test_ptm_sites.pkl')


import pickle 

# Extract sequences and PTM site labels from the reduced train and test DataFrames
train_sequences_reduced = reduced_train_df['Sequence'].tolist()
train_labels_reduced = reduced_train_df['PTM sites'].tolist()
test_sequences_reduced = reduced_test_df['Sequence'].tolist()
test_labels_reduced = reduced_test_df['PTM sites'].tolist()

# Save the lists to the specified pickle files
pickle_file_path = "50K_ptm_data/"

with open(pickle_file_path + "train_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_sequences_reduced, f)

with open(pickle_file_path + "test_sequences_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_sequences_reduced, f)

with open(pickle_file_path + "train_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(train_labels_reduced, f)

with open(pickle_file_path + "test_labels_chunked_by_family.pkl", "wb") as f:
    pickle.dump(test_labels_reduced, f)

# Return the paths to the saved pickle files
saved_files = [
    pickle_file_path + "train_sequences_chunked_by_family.pkl",
    pickle_file_path + "test_sequences_chunked_by_family.pkl",
    pickle_file_path + "train_labels_chunked_by_family.pkl",
    pickle_file_path + "test_labels_chunked_by_family.pkl"
]
saved_files



import pickle

def get_number_of_rows(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return len(data)

# Paths to the pickle files
files = [
    "50K_ptm_data/train_sequences_chunked_by_family.pkl",
    "50K_ptm_data/test_sequences_chunked_by_family.pkl",
    "50K_ptm_data/train_labels_chunked_by_family.pkl",
    "50K_ptm_data/test_labels_chunked_by_family.pkl"
]

# Get the number of rows for each file
number_of_rows = {file: get_number_of_rows(file) for file in files}
print(number_of_rows)