from transformers import AutoTokenizer, EsmForSequenceClassification
import torch
from pathlib import Path
import pandas as pd
import argparse
import gzip
from Bio import SeqIO
import re
import os
import xgboost as xgb
import time

def load_model(base_model_path, peft_model_path, only_raw_model=False):
    """
    Load the fine-tuned model and tokenizer from the given local path.
    """
    base_path = Path(base_model_path)
    model_path = Path(peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if not only_raw_model:
        model = EsmForSequenceClassification.from_pretrained(base_model_path, local_files_only=True)
        model.load_adapter(peft_model_path)
    else:
        model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer


def extract_embeddings(model, tokenizer, seq_list, seq_length=1024):
    if torch.cuda.is_available():
        model = model.base_model.cuda()
        model = torch.nn.DataParallel(model)
                
    seqs = seq_list
    inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
    
    with torch.no_grad():
        outputs = model(**inputs)
 
    last_hidden_states = outputs.last_hidden_state
    x = last_hidden_states.detach()
    x = x.mean(axis=1)
    return x


def create_fasta_iterator(fasta_file, chunk_size=1, info_filter=None, min_read_length=0):
    seqs = []
    record_ids = []
    record_info = []

    # Open the gzipped file
    with gzip.open(fasta_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Filter out records that don't have the specified info
            if (info_filter is not None and info_filter not in record.description) or len(record.seq) < min_read_length:
                continue
            else:
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


def find_all_fasta_files(directory, regex="^mgy_proteins_.*\.fa\.gz$"):
    compiled_regex = re.compile(regex)
    fasta_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if compiled_regex.search(file):
                fasta_files.append(os.path.join(root, file))
                
    return fasta_files


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


def main():
    parser = argparse.ArgumentParser(description="Run the protein sequence classification pipeline.")
    parser.add_argument("--peft_model_path", required=True, help="Path to the fine-tuned adapter model.")
    parser.add_argument("--base_model_path", required=True, help="Path to the base model.")
    parser.add_argument("--xgboost_model_path", required=True, help="Path to the XGBoost model.")
    parser.add_argument("--fasta_folder", required=True, help="Folder containing the FASTA files.")
    parser.add_argument("--output_folder", required=True, help="Folder to save the predictions.")
    parser.add_argument("--chunk_size", type=int, default=250, help="Number of sequences to process at once.")
    parser.add_argument("--min_seq_length", type=int, default=0, help="Minimum sequence length to consider.")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.base_model_path, args.peft_model_path)
    if args.fasta_folder.endswith("/"):
        fasta_files = find_all_fasta_files(args.fasta_folder)
    elif args.fasta_folder.endswith(".fa.gz"):
        fasta_files = [args.fasta_folder]
    
    xgb = XGBOOST_Classifier(args.xgboost_model_path)
    
    stt = time.time()
    print("Start time:", stt)
    
    for fasta_file in fasta_files:
        prediction_df = pd.DataFrame(columns=["record_id", "prediction", "info", "prediction_probability","sequence"])
        
        n = 0 
        for seqs, ids, info in create_fasta_iterator(fasta_file, chunk_size=args.chunk_size, info_filter="CR=1", min_read_length=args.min_seq_length):
            x = extract_embeddings(model, tokenizer, seqs, seq_length=1024).cpu().numpy()
            
            predictions = xgb.predict(x)
            prediction_probabilities = xgb.predict_proba(x)
            _df = {"record_id": ids, "prediction": predictions, "info": info, 
                   "prediction_probability": prediction_probabilities, "sequence": seqs}
            _df = pd.DataFrame.from_dict(_df, orient='index').T
            _df = _df[_df.prediction == 0]
            if _df.shape[0] > 0:
                prediction_df = pd.concat([prediction_df, _df], ignore_index=True)
            print("End time:", time.time() - stt)
            n = n + len(ids)
            if n % 1000 == 0:
                print(n)
                fasta_file = fasta_file.split("/")[-1]
                pandas_file = args.output_folder + fasta_file + "_predictions.csv"
                prediction_df.to_csv(pandas_file)
        
        fasta_file = fasta_file.split("/")[-1]
        pandas_file = args.output_folder + fasta_file + "_predictions.csv"
        prediction_df.to_csv(pandas_file)


if __name__ == "__main__":
    main()
