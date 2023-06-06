import torch
from torch.utils.data import Dataset

def init_text_datasets(train_df, dev_df):
    return DaicTextDataset(train_df), DaicTextDataset(dev_df)

class DaicTextDataset(Dataset):
    def __init__(self, df):
        self.df = df

        self.subject_ids = self.df['subject_id'].to_list()
        self.text_input = self.df['text_input'].to_list()
        self.binary_labels = self.df['PHQ8_Binary'].to_list()
        self.score_labels = self.df['PHQ8_Score'].to_list()

    def __getitem__(self, idx):
        text_input = self.text_input[idx]
        subject_id = self.subject_ids[idx]
        binary_label = torch.tensor(self.binary_labels[idx], dtype=torch.float32)
        score_label = torch.tensor(self.score_labels[idx], dtype=torch.float32)
    
        sample = {
            'subject_id': subject_id,
            'text_input': text_input, 
            'binary_label': binary_label,
            'score_label': score_label,
        }

        return sample

    def __len__(self):
        return len(self.subject_ids)
