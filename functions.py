import requests
from requests.adapters import HTTPAdapter, Retry
import re 
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# # #url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Ccc_interaction&format=tsv&query=Insulin%20AND%20%28reviewed%3Atrue%29&size=500'
#url = 'https://rest.uniprot.org/uniprotkb/search?fields=accession%2Cprotein_families%2Csequence%2Cft_mod_res%2Cabsorption%2Cft_act_site%2Cft_binding%2Ccc_catalytic_activity%2Ccc_cofactor%2Cft_dna_bind%2Cec%2Ccc_activity_regulation%2Ccc_function%2Ckinetics%2Ccc_pathway%2Cph_dependence%2Ctemp_dependence%2Cft_site%2Credox_potential%2Crhea%2Corganism_name&format=tsv&query=%28*%29&size=500'
url =  """https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Cprotein_families%2Csequence%2Cft_mod_res%2Cabsorption%2Cft_act_site%2Cft_binding%2Ccc_catalytic_activity%2Ccc_cofactor%2Cft_dna_bind%2Cec%2Ccc_activity_regulation%2Ccc_function%2Ckinetics%2Ccc_pathway%2Cph_dependence%2Ctemp_dependence%2Cft_site%2Credox_potential%2Crhea%2Corganism_name%2Cannotation_score%2Ckeyword%2Cgo%2Cgo_p%2Cgo_id%2Cgo_c%2Cgo_f&format=tsv&query=%28%28protein_name%3ACRISPR%29%29"""
# # get_uniprot_data(url, top_n = 10000,output_file='/home/salaris/protein_model/data/protein_functions/prot_func.tsv' )

class EmbeddingEncoder:
    """
        modelname = "sentence-transformers/all-MiniLM-L6-v2"
        modelname = 'tavakolih/all-MiniLM-L6-v2-pubmed-full'
    """
    def __init__(self, modelname = "sentence-transformers/all-MiniLM-L6-v2"):
        self.modelname = modelname
        self.model = SentenceTransformer(modelname)

    def encode(self, text):
        return self.model.encode(text)


class UniProtDataDownloader:
    """
    Example:
        downloader = UniProtDataDownloader(url, top_n=10000, output_file='/home/salaris/protein_model/data/protein_functions/prot_func.tsv')
        downloader.download_data()
        
    """
    def __init__(self, url, top_n=1000, output_file=None):
        self.url = url
        self.top_n = top_n
        self.output_file = output_file
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])

        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.output_columns = self.url.split('fields=')[1].split('&')[0].split('%2C')
        self.all_data = pd.DataFrame()

        # check if the file already exists:
        if self.output_file:
            if os.path.exists(self.output_file):
                exception = f"File {self.output_file} already exists. Please remove it or provide a different file name."
                raise Exception(exception)
            


    def get_next_link(self, headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')

        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def get_batch(self, batch_url):
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()
            total = response.headers["x-total-results"]
            yield response, total
            batch_url = self.get_next_link(response.headers)

    def download_data(self):
        entries_count = 0
        
        for batch, total in self.get_batch(self.url):
            rows_list = []

            for line in batch.text.splitlines()[1:]:
                tsv = line.split('\t')
                tsv_dict = {k: v for k, v in zip(self.output_columns, tsv)}
                rows_list.append(tsv_dict)

            entries_count += len(rows_list)
            print(f"Processed {entries_count} out of {total}")
            
            tmp_df = pd.DataFrame(rows_list)
            if self.output_file:
                tmp_df.to_csv(self.output_file, index=False, sep='\t', mode='a')
            else:
                self.all_data = pd.concat([self.all_data, tmp_df], ignore_index=True)

            if entries_count >= self.top_n:
                break

        if self.output_file:
            print(f"Data saved to {self.output_file}")
        else:
            return self.all_data
        

# downloader = UniProtDataDownloader(url, top_n=3000, output_file='/home/salaris/protein_model/function_prediction/data/prot_func.tsv')
# downloader.download_data()


#  output_model_file = /path/to/pytorch_mode.bin/
#  model_state_dict = torch.load(output_model_file) 
#  model = BertModel.from_pretrained(bert_model, state_dict=model_state_dict)


from transformers import AutoTokenizer, EsmForSequenceClassification
import torch
from pathlib import Path

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
        model = EsmForSequenceClassification.from_pretrained(base_model_path, local_files_only=True)
        # model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # print('loading the raw model')
        
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

def predict_sequence(model, tokenizer, sequence, max_length=512, to_cuda = True):
    """
    Predict the class of the given sequence using the fine-tuned model.
    """
    # Preprocess the input sequence
    inputs = tokenizer(sequence, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    if to_cuda:
        inputs.cuda()
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return logits, predictions.cpu().numpy()

