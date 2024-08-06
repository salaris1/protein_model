import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import pandas as pd
from datetime import datetime
import coolname
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from transformers import AutoTokenizer, TrainingArguments, EsmForMaskedLM
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
#-----------------for esm class:



torch.cuda.empty_cache()



os.environ["WANDB_NOTEBOOK_NAME"] = 'CAS_new.py'
# os.environ["WANDB_MODE"]= 'offline'


NUM_EPOCHS = 3
NUM_BATCHES = 8*4
EMB_FEATURES = 384 #Embedding counts 




root_folder = "/home/salaris/protein_model/function_prediction/"
datafolder = root_folder + "data/"
modelfolder = root_folder + "protein_model/model/"
modelname = "facebook/esm2_t6_8M_UR50D"
#modelname = "facebook/esm2_t30_150M_UR50D"
#modelname = "facebook/esm1b_t33_650M_UR50S"
# modelname= root_folder + "seq_models/esm2_t30_150M_UR50D_base_model_accelerated"




run = wandb.init(project='Prot_function_prediction')
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
df = pd.read_hdf(datafolder + "prot_func_with_embedding.h5", start = 0 , stop = 30000)

# Tokenization
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

def preprocess_data(examples, max_length=512):
    text = examples["sequence"]
    encoding = tokenizer(text, padding=True, truncation=True, max_length=max_length, is_split_into_words=False, add_special_tokens=False, return_tensors="pt")
    # Add regression labels
    encoding["regression_labels"] = torch.tensor([examples[f"e_{i}"] for i in range(EMB_FEATURES)], dtype=torch.float32).T
    return encoding

encoded_dataset = dataset.map(
    preprocess_data,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=dataset["train"].column_names,
)

encoded_dataset.set_format("torch")

# Model Loading and Setup for QLoRA
# Model Loading and Setup for QLoRA
class EsmWithRegression(EsmForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Add a regression head for 384 labels
        self.regression_head = nn.Linear(config.hidden_size, 384)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        # Extract last hidden state for regression
        last_hidden_state = outputs.hidden_states[-1]
        # Compute regression outputs
        regression_logits = self.regression_head(last_hidden_state[:, 0, :])  # Use the [CLS] token representation
        return outputs, regression_logits

model = EsmWithRegression.from_pretrained(modelname,output_hidden_states=True)






model.save_pretrained(base_modelfolder)

print_trainable_parameters(model)

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
    ]  # Modify as needed for your model
)

model = get_peft_model(model, lora_config)

print_trainable_parameters(model)

# Compute class weights due to class imbalance if necessary
# labels = np.array(df['class'].tolist())
# class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define the loss functions
# mlm_loss_fct = nn.CrossEntropyLoss(weight=class_weights.to('cuda'))
mlm_loss_fct = nn.CrossEntropyLoss()
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
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to='wandb',  # ensure that Weights & Biases is integrated
    save_total_limit=3,  # limit the total amount of checkpoints
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Prepare data loaders
train_dataloader = DataLoader(encoded_dataset['train'], batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(encoded_dataset['test'], batch_size=training_args.per_device_eval_batch_size, collate_fn=data_collator)

# Prepare optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

# Re-initialize the Accelerator
accelerator = Accelerator(device_placement=True)

print(f"Available GPUs: {torch.cuda.device_count()}")


# Prepare model, optimizer, and dataloaders for training with accelerator
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Manually train the model
for epoch in range(int(training_args.num_train_epochs)):
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        labels = batch['labels'].to(accelerator.device)
        regression_labels = batch['regression_labels'].to(accelerator.device)

        # Forward pass for MLM and regression
        outputs, regression_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Compute MLM loss
        mlm_loss = outputs.loss

        # Compute regression loss
        regression_loss = regression_loss_fct(regression_logits, regression_labels)

        # Combine losses (you may adjust the weights as needed)
        loss = mlm_loss + regression_loss

        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        if step % training_args.logging_steps == 0:
            print(f"Epoch {epoch + 1} Step {step}: Loss {loss.item()} (MLM: {mlm_loss.item()}, Regression: {regression_loss.item()})")
            wandb.log({"train_loss": loss.item(), "mlm_loss": mlm_loss.item(), "regression_loss": regression_loss.item(), "epoch": epoch + 1})


    # Evaluation
    model.eval()
    eval_loss = 0
    eval_steps = 0
    all_predictions = []
    all_labels = []

    for batch in eval_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)
            regression_labels = batch['regression_labels'].to(accelerator.device)

            outputs, regression_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            mlm_loss = outputs.loss
            regression_loss = regression_loss_fct(regression_logits, regression_labels)
            loss = mlm_loss + regression_loss
            eval_loss += loss.item()
            eval_steps += 1

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_loss /= eval_steps
    # Compute evaluation metrics (you can extend this to include regression metrics)
    metrics = {
        'eval_loss': eval_loss,
        'mlm_loss': mlm_loss.item(),
        'regression_loss': regression_loss.item()
    }
    print(f"Epoch {epoch + 1}: Evaluation Loss {eval_loss}, Metrics: {metrics}")
    wandb.log({"eval_loss": eval_loss, "eval_mlm_loss": mlm_loss.item(), "eval_regression_loss": regression_loss.item(), "epoch": epoch + 1})




# Save the fine-tuned model
model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save the model as a PyTorch model
torch_model_path = os.path.join(modelfolder, "esm_finetuned_accelerated", run.name,"model.pth")
torch.save(model.state_dict(), torch_model_path)
print(f"Model state dict saved to {torch_model_path}")

# Optionally, evaluate the model on the test set
results = metrics
print(results)
