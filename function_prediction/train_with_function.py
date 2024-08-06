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
from transformers import EsmForMaskedLM
from datasets import Dataset
from accelerate import Accelerator, DistributedType
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import pickle
from transformers import DataCollatorForLanguageModeling
from transformers import EsmForSequenceClassification

torch.cuda.empty_cache()
import coolname 

# Initialize accelerator
accelerator = Accelerator()

# Set the CUDA visible devices based on the accelerator
if accelerator.distributed_type == DistributedType.MULTI_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

os.environ["WANDB_NOTEBOOK_NAME"] = 'CAS_new.py'
os.environ["WANDB_MODE"]= 'offline'


NUM_EPOCHS = 3
NUM_BATCHES = 8


root_folder = "/home/salaris/protein_model/function_prediction/"
datafolder = root_folder + "data/"
modelfolder = root_folder + "protein_model/model/"
modelname = "facebook/esm2_t6_8M_UR50D"
#modelname = "facebook/esm2_t30_150M_UR50D"
#modelname = "facebook/esm1b_t33_650M_UR50S"
# modelname= root_folder + "seq_models/esm2_t30_150M_UR50D_base_model_accelerated"



run = wandb.init(project='protein_function_prediction')
run_name  = run.name
if run_name == '':
    run_name = coolname.generate_slug(2)

print("run.name: --> ", run.name)

if root_folder in modelname:
    base_modelfolder = modelname
else:
    base_modelfolder = modelfolder + modelname.split("/")[-1] +"_base_model_accelerated/"

print("base_modelfolder: ", base_modelfolder)


# Read and preprocess data
df = pd.read_hdf(datafolder + "prot_func_with_embedding.h5", start =0 , stop = 2000)
df['class'] = df['total_embedding'].astype('float')


# Tokenization
modelname_str = modelname.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(modelname)

max_sequence_length = 512

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

def preprocess_data(examples, max_length):
    text = examples["sequence"]
    encoding = tokenizer(text, padding=True, truncation=True, max_length=max_length, is_split_into_words=False, add_special_tokens=False, return_tensors="pt")
    encoding["labels"] = examples["class"]
    return encoding

encoded_dataset = dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset["train"].column_names,
    fn_kwargs={"max_length": max_sequence_length}

)

encoded_dataset.set_format("torch")

# Model Loading and Setup for QLoRA
from transformers import EsmModel, EsmConfig

# Assuming you need to customize the last layer for regression
# class EsmForSequenceRegression(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.esm = EsmModel.from_pretrained(modelname)
#         self.regressor = nn.Linear(config.hidden_size, 1)  # Change output size to 1 for regression

#     def forward(self, input_ids, attention_mask=None):
#         outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = outputs.last_hidden_state[:, 0, :]  # Assuming using the [CLS] token
#         logits = self.regressor(sequence_output)
#         return logits
#     def save_pretrained(self, path):
#         self.esm.save_pretrained(path)
#         torch.save(self.regressor.state_dict(), os.path.join(path, "regressor.pth"))

# config = EsmConfig.from_pretrained(modelname)
# model = EsmForSequenceRegression(config)

# Model Loading and Setup for QLoRA
model = EsmForSequenceClassification.from_pretrained(modelname, num_labels=1)


model.save_pretrained(base_modelfolder)

# model = prepare_model_for_kbit_training(model)

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

# # Compute class weights due to class imbalance if necessary
# # labels = np.array(encoded_dataset['train']['labels'])
# labels = np.array(df['class'].tolist())
# class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define the loss function to incorporate class weights
# loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))

loss_fct = nn.MSELoss()

# Trainer Configuration
training_args = TrainingArguments(
    output_dir=modelfolder,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=NUM_BATCHES,
    per_device_eval_batch_size=NUM_BATCHES,
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
    save_total_limit=3,  # limit the total amount of checkpoints
)

# Define metrics for evaluation
from torch.nn.functional import cross_entropy

from sklearn.metrics import mean_squared_error, r2_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    r2 = r2_score(labels, logits)
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=True,
#     mlm_probability=0.15
# )

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels").float()  # Ensure labels are float for regression
#         outputs = model(**inputs)
#         logits = outputs.view(-1)
#         loss = loss_fct(logits, labels)
#         return (loss, outputs) if return_outputs else loss


# Initialize the Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Check if there's a checkpoint to resume from
checkpoint_dir = os.path.join(modelfolder, 'checkpoint')
if os.path.exists(checkpoint_dir):
    last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]  # Get the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
else:
    checkpoint_path = None


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
trainer.train(resume_from_checkpoint=checkpoint_path)

# Save the fine-tuned model
model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save the model as a PyTorch model
torch_model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name,"model.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"Model state dict saved to {torch_model_path}")

# Optionally, evaluate the model on the test set
results = trainer.evaluate()
print(results)
