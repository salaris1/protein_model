import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator as DataCollatorForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from transformers import EsmForSequenceClassification
from datasets import Dataset
from accelerate import Accelerator, DistributedType
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import pickle

# Initialize accelerator
accelerator = Accelerator()

# Set the CUDA visible devices based on the accelerator
if accelerator.distributed_type == DistributedType.MULTI_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

os.environ["WANDB_NOTEBOOK_NAME"] = 'CAS_new.py'
wandb.init(project='CAS_classification')

datafolder = "/home/salaris/protein_model/data/"
modelfolder = "/home/salaris/protein_model/models/"
pickle_file_path = datafolder + "cas_data_512_v1/"  # --> where to save the files 

# Read and preprocess data
df = pd.read_csv(datafolder + "all_data_20240629_09.csvtrain_test.csv", sep="\t", nrows=6000)
df['class'] = df['class'].astype('category')
df['class'] = df['class'].cat.codes

# Tokenization
modelname = "facebook/esm2_t6_8M_UR50D"
modelname_str = modelname.split("/")[1]
tokenizer = AutoTokenizer.from_pretrained(modelname)

max_sequence_length = 1024

dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, shuffle=True)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def preprocess_data(examples, max_length=512):
    text = examples["seq"]
    encoding = tokenizer(text, padding=True, truncation=True, max_length=max_length, is_split_into_words=False, add_special_tokens=False, return_tensors="pt")
    encoding["labels"] = examples["class"]
    return encoding

encoded_dataset = dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset["train"].column_names,
)

encoded_dataset.set_format("torch")

# Model Loading and Setup for QLoRA
model = EsmForSequenceClassification.from_pretrained(modelname, num_labels=13)
model = prepare_model_for_kbit_training(model)

print_trainable_parameters(model)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=[
        "query",
        "key",
        "value",
        "EsmSelfOutput.dense",
        "EsmIntermediate.dense",
        "EsmOutput.dense",
        "EsmContactPredictionHead.regression",
        "EsmClassificationHead.dense",
        "EsmClassificationHead.out_proj",
    ]  # Modify as needed for your model
)

model = get_peft_model(model, lora_config)

print_trainable_parameters(model)

# Compute class weights due to class imbalance if necessary
labels = np.array(encoded_dataset['train']['labels'])
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define the loss function to incorporate class weights
loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))

# Modify compute_loss in Trainer to use the custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Trainer Configuration
training_args = TrainingArguments(
    output_dir=modelfolder,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to='wandb',  # ensure that Weights & Biases is integrated
)

# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Prepare model, optimizer, and dataloaders for training with accelerator

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    trainer.model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
)

# Train the model
trainer.train()

# Save the fine-tuned model
model_path = os.path.join(modelfolder, "esm_finetuned")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Optionally, evaluate the model on the test set
results = trainer.evaluate()
print(results)
