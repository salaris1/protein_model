import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


pickle_file_path = "all_ptm_data_512/"


# Return the paths to the saved pickle files
saved_files = [
    pickle_file_path + "train_sequences_chunked_by_family.pkl",
    pickle_file_path + "test_sequences_chunked_by_family.pkl",
    pickle_file_path + "train_labels_chunked_by_family.pkl",
    pickle_file_path + "test_labels_chunked_by_family.pkl"
]
saved_files

#%%

import wandb
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import pickle
#%%
# Initialize accelerator and Weights & Biases
accelerator = Accelerator()

# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_NOTEBOOK_NAME"] = 'new.py'
wandb.init(project='ptm_site_prediction') #disabling this 
# wandb.init(mode = 'disabled')



import transformers
import logging

transformers.logging.set_verbosity_error()  # Set transformers to only log errors
logging.getLogger("pytorch").setLevel(logging.ERROR)  # Set PyTorch to only log errors


#%%

# Helper Functions and Data Preparation
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def save_config_to_txt(config, filename):
    """Save the configuration dictionary to a text file."""
    with open(filename, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def truncate_labels(labels, max_length):
    return [label[:max_length] for label in labels]

 

def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    bacc = balanced_accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc, 'bacc': bacc}

def compute_loss(model, logits, inputs):
    # logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    # active_logits = logits.view(-1, model.config.num_labels)
    active_logits = logits.view(-1,2)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

#%%

# Load data from pickle files

with open("2100K_ptm_data_512/train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)
    
with open("2100K_ptm_data_512/test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)

with open("2100K_ptm_data_512/train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)

with open("2100K_ptm_data_512/test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)

#%%

# Tokenization
# modelname = "facebook/esm2_t30_150M_UR50D"
modelname = "facebook/esm2_t6_8M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(modelname)

# Set max_sequence_length to the tokenizer's max input length
max_sequence_length = 1024

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False, add_special_tokens=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False, add_special_tokens=False)

# Directly truncate the entire list of labels
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)

#%%

#%%
train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)



# Compute Class Weights
classes = np.array([0, 1])  
flat_train_labels = np.array([label for sublist in train_labels for label in sublist])
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)


#%%

# Define Custom Trainer Class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
  
        logits = outputs.logits
        loss = compute_loss(model, logits, inputs)
        return (loss, outputs) if return_outputs else loss

#%%

# Configure the quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#%%

def train_function_no_sweeps(train_dataset, test_dataset):
    
    # Directly set the config
    config = {
        "lora_alpha": 1, 
        "lora_dropout": 0.5,
        "lr": 3.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 15,
        "per_device_train_batch_size": 36,
        "r": 1, # was 2
        "weight_decay": 0.3,
        # Add other hyperparameters as needed
    }

    # Log the config to W&B
    wandb.config.update(config)

    # Save the config to a text file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_filename = f"esm2_t30_150M_qlora_ptm_config_{timestamp}.txt"
    save_config_to_txt(config, config_filename)
    
        
    model_checkpoint = modelname # "facebook/esm2_t30_150M_UR50D"  
    
    # Define labels and model

    id2label = {0: "No ptm site", 1: "ptm site"}
    label2id = {v: k for k, v in id2label.items()}
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config  # Apply quantization here
    )

    # Prepare the model for 4-bit quantization training
    model.gradient_checkpointing_enable()
    # model.gradient_checkpointing_disable()
    model = prepare_model_for_kbit_training(model)
    
    # Convert the model into a PeftModel
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=[
            "query",
            "key",
            "value",
            "EsmSelfOutput.dense",
            "EsmIntermediate.dense",
            "EsmOutput.dense",
            "EsmContactPredictionHead.regression",
            "classifier"
        ],
        lora_dropout=config["lora_dropout"],
        bias="none",  # or "all" or "lora_only"
        # modules_to_save=["classifier"]
    )

    print_trainable_parameters(model) # added this in
    model = get_peft_model(model, peft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Training setup
    training_args = TrainingArguments(
        output_dir=f"esm2_t30_150M_qlora_ptm_sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1, # changed from 1 to 4
        # warmup_steps=2, # added this in 
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=3,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb', 
     
        optim="paged_adamw_8bit" # added this in 

    )
    
    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and Save Model
    trainer.train()
    save_path = os.path.join("qlora_ptm_sites", f"best_model_esm2_t30_150M_qlora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)


#%%


# Call the training function
if __name__ == "__main__":
    _ = train_function_no_sweeps(train_dataset, test_dataset)



# %%
