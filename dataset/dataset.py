from os import listdir
from os.path import isfile, join
import pandas as pd
from natsort import natsorted

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
