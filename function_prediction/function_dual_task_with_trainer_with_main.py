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
import argparse

def main(
    root_folder,
    datafolder,
    modelfolder,
    modelname,
    num_epochs,
    num_batches,
    emb_features,
    log_sigma_mlm,
    log_sigma_regression,
    wandb_project
):
    torch.cuda.empty_cache()

    os.environ["WANDB_NOTEBOOK_NAME"] = 'CAS_new.py'

    log_sigma_mlm = nn.Parameter(torch.tensor(log_sigma_mlm))
    log_sigma_regression = nn.Parameter(torch.tensor(log_sigma_regression))

    run = wandb.init(project= wandb_project)
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
            [examples[f"e_{i}"] for i in range(emb_features)], dtype=torch.float32
        ).T
        encoding["labels"] = encoding["input_ids"].clone()  # Add labels for MLM
        return encoding

    encoded_dataset = dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    )

    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'regression_labels'])

    class EsmWithRegression(EsmForMaskedLM):
        def __init__(self, config):
            super().__init__(config)
            self.regression_head = nn.Linear(config.hidden_size, emb_features)
        
        def forward(self, input_ids=None, attention_mask=None, labels=None, regression_labels=None, **kwargs):
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            last_hidden_state = outputs.hidden_states[-1]
            mean_pooled = last_hidden_state[:, 0, :]
            regression_logits = self.regression_head(mean_pooled)
            self.log_sigma_mlm = log_sigma_mlm
            self.log_sigma_regression = log_sigma_regression

            # MLM Loss
            mlm_loss = F.cross_entropy(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100)

            # Regression Loss
            regression_loss = F.mse_loss(regression_logits, regression_labels, reduction='mean')

            # Combine losses with uncertainty weights
            loss = self.log_sigma_mlm * mlm_loss + \
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
    mlm_loss_fct = F.cross_entropy
    regression_loss_fct = nn.MSELoss()

    training_args = TrainingArguments(
        output_dir=modelfolder,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=num_batches,
        per_device_eval_batch_size=num_batches,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=250,
        save_steps=1000,
        load_best_model_at_end=True,
        greater_is_better=True,
        report_to='wandb',
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    def compute_metrics(eval_pred):
        # Unpack the predictions and labels
        (logits, regression_logits), (labels, regression_labels) = eval_pred

        # Ensure tensors are correctly typed
        logits = torch.tensor(logits, dtype=torch.float32)
        regression_logits = torch.tensor(regression_logits, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        regression_labels = torch.tensor(regression_labels, dtype=torch.float32)

        # Compute MLM loss
        mlm_loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1), ignore_index=-100)

        # Compute regression loss
        regression_loss = regression_loss_fct(regression_logits, regression_labels)

        # Combine losses (you may adjust the weights as needed)
        loss = log_sigma_mlm * mlm_loss + \
            log_sigma_regression * regression_loss 
        
        # Compute additional metrics as needed
        metrics = {
            'loss': loss.item(),
            'mlm_loss': mlm_loss.item(),
            'regression_loss': regression_loss.item(),
        }

        wandb.log(metrics)
        return metrics

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Function Prediction Model Training")

    parser.add_argument("--root_folder", type=str, required=True, help="Root folder for the project")
    parser.add_argument("--datafolder", type=str, required=True, help="Data folder path")
    parser.add_argument("--modelfolder", type=str, required=True, help="Model folder path")
    parser.add_argument("--modelname", type=str, required=True, help="Model name or path")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_batches", type=int, default=8, help="Number of batches per device")
    parser.add_argument("--emb_features", type=int, default=384, help="Number of embedding features")
    parser.add_argument("--wandb_project", type=str, default="Prot_function_prediction", help="Wandb project name")
    parser.add_argument("log_sigma_mlm", type=float, default=0.0, help="Log sigma for MLM loss")
    parser.add_argument("log_sigma_regression", type=float, default=100.0, help="Log sigma for regression loss")    
    args = parser.parse_args()

    main(
        root_folder=args.root_folder,
        datafolder=args.datafolder,
        modelfolder=args.modelfolder,
        modelname=args.modelname,
        num_epochs=args.num_epochs,
        num_batches=args.num_batches,
        emb_features=args.emb_features,
        log_sigma_mlm=args.log_sigma_mlm,
        log_sigma_regression=args.log_sigma_regression,
        wandb_project=args.wandb_project

    )
