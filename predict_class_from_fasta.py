from transformers import AutoTokenizer, EsmForSequenceClassification
import torch
from pathlib import Path
import pandas as pd 
def load_model(base_model_path, peft_model_path,only_raw_model = False):
    """
    Load the fine-tuned model and tokenizer from the given local path.
    """
    base_path = Path(base_model_path)
    model_path = Path(peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if not only_raw_model:
        # method1
        #model_state_dict = torch.load(f"{peft_model_path}/model.pth")
        #model = EsmForSequenceClassification.from_pretrained(base_model_path, local_files_only=True,state_dict = model_state_dict)

        #method 2
        model = EsmForSequenceClassification.from_pretrained(base_model_path, local_files_only=True)
        model.load_adapter(peft_model_path)
        # print('loading the finetuned model')
    else:
        # model = EsmForSequenceClassification.from_pretrained(base_model_path, local_files_only=True)
        model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # print('loading the raw model')
        
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer


def extract_embeddings(model,tokenizer, seq_list, seq_length = 1024):
    from transformers import EsmTokenizer, EsmModel
    import torch
    if torch.cuda.is_available():
        model = model.base_model.cuda()
        model = torch.nn.DataParallel(model)
                
    seqs =seq_list
    inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True,max_length = seq_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
 
    last_hidden_states = outputs.last_hidden_state
    x = last_hidden_states.detach()
    x= x.mean(axis=1)
    return(x)

import gzip
from Bio import SeqIO

def create_fasta_iterator(fasta_file, chunk_size=1, info_filter = None):
    seqs = []
    record_ids = []
    record_info = []

    # Open the gzipped file
    with gzip.open(fasta_file, "rt") as handle:

        for record in SeqIO.parse(handle, "fasta"):
            # Filter out records that don't have the specified info
            if info_filter is not None and info_filter not in record.description:
                continue
            seqs.append(str(record.seq))
            record_ids.append(record.id)
            record_info.append(record.description)

            if len(seqs) == chunk_size:
                yield seqs, record_ids, record_info
                seqs = []
                record_ids = []
                record_info = []

        # Yield any remaining sequences
        if len(seqs) > 0:
            yield seqs, record_ids, record_info




import re
import os

def find_all_fasta_files(directory, regex="^mgy_proteins_.*\.fa\.gz$"):
    compiled_regex = re.compile(regex)
    fasta_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if compiled_regex.search(file):
                fasta_files.append(os.path.join(root, file))
                
    return fasta_files


import xgboost as xgb

class XGBOOST_Classifier:
    def __init__(self, xgboost_model_path=None):
        self.model = xgb.XGBClassifier()
        if xgboost_model_path:
            self.model.load_model(xgboost_model_path)
        else:
            raise ValueError("A valid path to the xgboost model must be provided.")
    
    def predict(self, embeddings):
        preds = self.model.predict(embeddings)
        return preds
    
    def predict_proba(self, embeddings):
        preds = self.model.predict_proba(embeddings)
        return preds

peft_model_path = "/home/salaris/protein_model/seq_models/esm_finetuned_accelerated/jolly-capybara-19/"
base_model_path = "/home/salaris/protein_model/seq_models/esm2_t30_150M_UR50D_base_model_accelerated/"
xgboost_model_path = "/home/salaris/protein_model/xgboost_model/150M_xgboost_model.json"
fasta_folder = "/home/salaris/protein_model/esm30_data/"

def main():
    model, tokenizer = load_model(base_model_path, peft_model_path)
    fasta_files = find_all_fasta_files(fasta_folder)
    #initialize xgboost:
    xgb = XGBOOST_Classifier(xgboost_model_path)
    import time
    stt = time.time()
    print("start time:",stt)
    for fasta_file in fasta_files:
        prediction_df = pd.DataFrame(columns = ["record_id", "prediction", "info", "prediction_probability"])
        # embeddings, record_ids, record_info = extract_embeddings_from_fasta_file(fasta_file, model, tokenizer)
        n = 0 
        for seqs, ids, info in create_fasta_iterator(fasta_file,chunk_size=250, info_filter="CR=1"):
            x = extract_embeddings(model,tokenizer, seqs, seq_length= 1024).cpu().numpy()
            
            predictions = xgb.predict(x)
            prediction_probabilities = xgb.predict_proba(x)
            _df = {"record_id": ids, "prediction": predictions, "info": info, "prediction_probability": prediction_probabilities}
            _df = pd.DataFrame.from_dict(_df, orient='index').T
            _df = _df[_df.prediction == 0 ]
            if _df.shape[0]>0:
                prediction_df = pd.concat([prediction_df,_df], ignore_index=True)
            # print(prediction_df.head())
            print("end time:" , time.time() - stt )
            n = n+ len(ids)
            if n % 1000 == 0 :
                print(n)
        pandas_File = fasta_file + "_predictions.csv"
        prediction_df.to_csv(pandas_File)
        


if __name__ == "__main__":
    main()



    

