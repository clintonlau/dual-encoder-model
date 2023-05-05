from os import listdir
from os.path import isfile, join
import pandas as pd
from natsort import natsorted
import torch
from torch.utils.data import Dataset

def create_input_df(dir, label_data):
    """
    Join transcripts in input partition-specific directory and label data, return joined dataframe
    """
    file_paths = natsorted([filename for filename in listdir(dir) if isfile(join(dir, filename))])
    subject_ids, text_inputs, resp_timestamps = transcripts_to_list(file_paths)

    transcript_df = pd.DataFrame({
        'subject_id': subject_ids,
        'text_input': text_inputs,
        'response_timestamps': resp_timestamps,
    })
    return pd.merge(transcript_df, label_data, on=['subject_id'])

def transcripts_to_list(file_paths):
    """
    Read file paths to transcripts, return lists of subject ids, text inputs, and response timestamps
    """
    subject_ids, text_inputs, resp_timestamps = [], [], []

    for file_path in file_paths:
        subject_id = int(file_path.split('_')[0])
        transcript_df = pd.read_csv(join(dir, file_path), sep='\t')

        ellie_rows = transcript_df.iloc[::2]
        assert all(ellie_rows['speaker'] == 'Ellie')
        subject_rows = transcript_df.iloc[1::2]
        assert all(subject_rows['speaker'] == 'Participant')

        ellie_prompts = ellie_rows['value'].tolist()
        subject_responses = subject_rows['value'].tolist()
        response_start_time = subject_rows['start_time'].tolist()
        response_stop_time = subject_rows['stop_time'].tolist()
        response_start_time = pd.to_numeric(subject_rows['start_time']).tolist()
        response_stop_time = pd.to_numeric(subject_rows['stop_time']).tolist()
        response_timestamp = [(start_time, stop_time) for start_time, stop_time in zip(response_start_time, response_stop_time)]

        ellie_prompts = [prompt + str('?') for prompt in ellie_prompts]
        subject_responses = [response + str('.') for response in subject_responses]
        assert len(ellie_prompts) == len(subject_responses)
        
        interview_sentences = [prompt+' '+response for prompt, response in zip(ellie_prompts, subject_responses)]
        
        subject_ids.append(subject_id)
        text_inputs.append(interview_sentences)
        resp_timestamps.append(response_timestamp)

    return subject_ids, text_inputs, resp_timestamps

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
