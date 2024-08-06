import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
import coolname
from transformers import AutoTokenizer, TrainingArguments, EsmForMaskedLM, Trainer
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import torch.nn.functional as F
torch.cuda.empty_cache()

os.environ["WANDB_NOTEBOOK_NAME"] = 'CAS_new.py'

NUM_EPOCHS = 5
NUM_BATCHES = 8*1
EMB_FEATURES = 384  # Embedding counts

log_sigma_mlm = nn.Parameter(torch.tensor(0.0))
log_sigma_regression = nn.Parameter(torch.tensor(100.0))


root_folder = "/home/salaris/protein_model/function_prediction/"
datafolder = root_folder + "data/"
modelfolder = root_folder + "protein_model/model/"
modelname = "facebook/esm2_t6_8M_UR50D"

run = wandb.init(project='Prot_function_prediction')
run_name = run.name
if run_name == '':
    run_name = coolname.generate_slug(2)

print("run.name: --> ", run.name)

if root_folder in modelname:
    base_modelfolder = modelname
else:
    base_modelfolder = modelfolder + modelname.split("/")[-1] + "_base_model_accelerated/"

print("base_modelfolder: ", base_modelfolder)

# Read and preprocess data
df = pd.read_hdf(datafolder + "prot_func_with_embedding2.h5", start=0, stop=300000)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(modelname)

max_sequence_length = 512

dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, shuffle=True)

def preprocess_data(examples, max_length=512):
    text = examples["sequence"]
    encoding = tokenizer(
        text, padding=True, truncation=True, max_length=max_length,
        is_split_into_words=False, add_special_tokens=False, return_tensors="pt"
    )
    # Add regression labels
    encoding["regression_labels"] = torch.tensor(
        [examples[f"e_{i}"] for i in range(EMB_FEATURES)], dtype=torch.float32
    ).T
    encoding["labels"] = encoding["input_ids"].clone()  # Add labels for MLM
    return encoding

encoded_dataset = dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset["train"].column_names,
)

# encoded_dataset.set_format("torch")
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'regression_labels'])


# Model Loading and Setup for QLoRA
# class EsmWithRegression(EsmForMaskedLM):
#     def __init__(self, config):
#         super().__init__(config)
#         # Add a regression head for 384 labels
#         self.regression_head = nn.Linear(config.hidden_size, EMB_FEATURES)

#     def forward(self, input_ids=None, attention_mask=None, labels=None, regression_labels=None, **kwargs):
#         outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
#         # Extract last hidden state for regression
#         last_hidden_state = outputs.hidden_states[-1]
#         pooled_embedding = last_hidden_state[:, 0, :]
#         # pooled_embedding = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
#         # Compute regression outputs
#         regression_logits = self.regression_head(pooled_embedding)  # Use the [CLS] token representation
#         return outputs.loss, outputs.logits, regression_logits

import torch.nn.functional as F

class EsmWithRegression(EsmForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.regression_head = nn.Linear(config.hidden_size, EMB_FEATURES)
       

    def forward(self, input_ids=None, attention_mask=None, labels=None, regression_labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        last_hidden_state = outputs.hidden_states[-1]
        # mean_pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        mean_pooled = last_hidden_state[:, 0, :]
        regression_logits = self.regression_head(mean_pooled)
        self.log_sigma_mlm = log_sigma_mlm
        self.log_sigma_regression = log_sigma_regression

        # MLM Loss
        mlm_loss = F.cross_entropy(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

        # Regression Loss
        regression_loss = F.mse_loss(regression_logits, regression_labels, reduction='mean')

        # Combine losses with uncertainty weights
        loss = self.log_sigma_mlm* mlm_loss + \
               self.log_sigma_regression * regression_loss 
        
        return {
            "loss": loss,
            "logits": outputs.logits,
            "regression_logits": regression_logits,
            "regression_labels": regression_labels
        }


model = EsmWithRegression.from_pretrained(modelname, output_hidden_states=True)
model.save_pretrained(base_modelfolder)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
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
    ]
)

model = get_peft_model(model, lora_config)

# Define the loss functions
# mlm_loss_fct = nn.CrossEntropyLoss()
mlm_loss_fct = F.cross_entropy

regression_loss_fct = nn.MSELoss()

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
    eval_steps=250,
    save_steps=1000,
    load_best_model_at_end=True,
    # metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to='wandb',
    save_total_limit=3,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    # Unpack the predictions and labels
    (logits, regression_logits), (labels, regression_labels) = eval_pred

    # Ensure tensors are correctly typed
    logits = torch.tensor(logits, dtype=torch.float32)
    regression_logits = torch.tensor(regression_logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    regression_labels = torch.tensor(regression_labels, dtype=torch.float32)

    # Compute MLM loss
    # mlm_loss = mlm_loss_fct(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
    mlm_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100)

    # Compute regression loss
    regression_loss = regression_loss_fct(regression_logits, regression_labels)

    # Combine losses (you may adjust the weights as needed)
    # loss = mlm_loss + regression_loss
    loss = log_sigma_mlm* mlm_loss + \
        log_sigma_regression * regression_loss 
    # Compute additional metrics as needed
    metrics = {
        'loss': loss.item(),
        'mlm_loss': mlm_loss.item(),
        'regression_loss': regression_loss.item(),
    }

    wandb.log(metrics)
    return metrics


from transformers import Trainer, EvalPrediction

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Prepare inputs for the model
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Forward pass through the model
            outputs = model(**inputs)
            loss, logits, regression_logits = outputs['loss'], outputs['logits'], outputs['regression_logits']

        # Extract labels from inputs
        labels = inputs.get('labels')
        regression_labels = inputs.get('regression_labels')

        # If only loss is required, return early
        if prediction_loss_only:
            return (loss, None, None)

        # Return EvalPrediction with both types of outputs
        return (loss, (logits, regression_logits), (labels, regression_labels))

# Instantiate the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# Train the model
trainer.train()

# Save the fine-tuned model
model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save the model as a PyTorch model
torch_model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name, "model.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"Model state dict saved to {torch_model_path}")
