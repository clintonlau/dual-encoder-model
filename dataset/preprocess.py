import os
import pandas as pd
from tqdm.notebook import tqdm
import more_itertools as mit

PARTICIPANT_DATA_DIR = '../raw-data'


# files with interruptions in seconds
# dictionary - subject_id - timestamped onset/end time
interrupt = {
    373: [398.0, 430.3],
    444: [285.6, 384.4]
}

# files with misaligned transcripts (add these values to all the timestamps)
# dictionary - subject_id - misalignment in seconds
misaligned = {
    318: 34.319917,
    321: 3.8379167,
    341: 6.1892,
    362: 16.8582
}

START_PROMPTS = "are you okay with this|hi i'm ellie thanks for coming in today|think of me as a friend i don't judge|and please feel free to tell me anything|i'm not a therapist|i'll ask a few questions|i was created to talk"
CLOSING_PROMPTS = "okay i think i have asked everything i need to|okay i think i've asked everything i need to|goodbye|it was great chatting with you|thanks for sharing your thoughts"

def remove_unique_identifiers(df):
    for index, row in df.iterrows():
        if row['speaker'] == 'Ellie' and '(' in df.loc[index,'value']:
            df.loc[index,'value'] = df.loc[index,'value'].split('(')[-1].strip(')') # keep content in parentheses

def remove_whitespace(df):
    df['value'] = df['value'].str.strip()
    df['value'] = df['value'].str.replace(' +', ' ')

def remove_annotations(df):
    df.drop(df[df['value'].str.contains('<sync>|<sync.|<synch>|\[syncing\]|\[sync\]|\[synch\]|\[synching\]') == True].index, inplace=True)
    df['value'] = df['value'].str.replace('<laughter>', '*laughter*')
    df['value'] = df['value'].str.replace('\[laughter\]', '*laughter*')
    df['value'] = df['value'].str.replace('<sigh>', '*sigh*')        
    df['value'] = df['value'].str.replace('<.*?>', '', regex=True) # remove all <> instances
    df['value'] = df['value'].str.replace('\[.*?\]', '', regex=True) # remove all [] instances
    df['value'] = df['value'].str.replace('scrubbed_entry', '')
    df['value'] = df['value'].str.replace('xxxx', '')
    df['value'] = df['value'].str.replace('xxx', '')

def remove_empty_rows(df):
    df = df[df['value'] != '']
    df = df[df['value'] != 'laughter']
    df = df[df['value'] != '*laughter*']
    df = df.reset_index(drop=True)

def collapse_responses(df):
    # grouping dataframe index by consecutive numbers 
    participant_index = list(df.loc[df['speaker'] == 'Participant'].index)
    participant_index = [list(group) for group in mit.consecutive_groups(participant_index)]
    # only keep groups that have multiple lines to be collapsed
    participant_index_group = [i for i in participant_index if len(i) > 1]

    for group in participant_index_group:
        first_line_index = group[0]
        last_line_index = group[-1]

        start_time = df.loc[first_line_index]['start_time']
        stop_time = df.loc[last_line_index]['stop_time']
        value = ', '.join([df.loc[i]['value'] for i in group])
        df.loc[first_line_index] = [start_time, stop_time, 'Participant', value]
        df.drop(group[1:], axis=0, inplace=True)

def to_qr_pairs(df):
        df = df.reset_index(drop=True)
        lst_of_participant_index = df[df['speaker'] == 'Participant'].index.tolist()
        # remove any 0's from list (where Participant starts)
        first_ellie_idx = df[df['speaker'] == 'Ellie'].index.tolist()[0]
        count = 0 
        while count < first_ellie_idx:
            lst_of_participant_index.pop(0)
            count += 1
        # insert index-1 to the list
        lst_of_qr_index = []
        # takes Ellie's prompt associated with the response
        for e in lst_of_participant_index:
            lst_of_qr_index.append(e-1)
            lst_of_qr_index.append(e)  
        return df.loc[lst_of_qr_index,:]

def preprocess_transcripts(ids, output_dir) -> None:
    for id in tqdm(ids):
    
        id_dir = os.path.join(PARTICIPANT_DATA_DIR, str(id)+'_P')
        transcript = pd.read_csv(os.path.join(id_dir, str(id)+'_TRANSCRIPT.csv'), sep='\t')

        if id in interrupt:
            onset_time, end_time = interrupt[id]
            onset_idx = transcript[transcript['start_time'].astype(str).str.contains(str(onset_time))].index.values[0]
            end_idx = transcript[transcript['stop_time'].astype(str).str.contains(str(end_time))].index.values[0]
            transcript.drop(transcript.index[onset_idx:end_idx], inplace=True)
        
        if id in misaligned:
            offset_value = misaligned[id]
            transcript['start_time'] = transcript['start_time'] + offset_value
            transcript['stop_time'] = transcript['stop_time'] + offset_value

        transcript.dropna(inplace=True)
        remove_annotations(transcript)
        remove_whitespace(transcript)
        remove_unique_identifiers(transcript)
        remove_empty_rows(transcript)
        collapse_responses(transcript)

        transcript.drop(transcript[transcript['value'].str.contains(START_PROMPTS) == True].index, inplace=True)
        transcript.drop(transcript[transcript['value'].str.contains(CLOSING_PROMPTS) == True].index, inplace=True)

        transcript['value'] = transcript['value'].str.replace('_','') # remove '_' which are for acronyms
        transcript['value'] = transcript['value'].str.lower()

        qr_transcript = to_qr_pairs(transcript)
        qr_transcript.to_csv(os.path.join(output_dir, str(id) + '_TRANSCRIPT_cleaned.csv'), sep='\t', index=False)