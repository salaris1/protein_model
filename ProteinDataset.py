from torch.utils.data import Dataset, DataLoader
import torch


class ProteinDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["seq"]
        label = row["class"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(label)
        return encoding
