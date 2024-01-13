import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from dataset.dataset import init_text_datasets

from model.utils import config_run
from model.dual_encoder import init_dual_encoder, DualEncoderCollator


config_path = '/config/dual_encoder.yaml'
cfg, hparam, device = config_run(config_path)
loss_function = torch.nn.MSELoss()

model = init_dual_encoder(hparam)
optimizer = optim.Adam(model.parameters(), lr=hparam.learning_rate)
collator = DualEncoderCollator( hparam.transformer_pretrained_id
                               , hparam.st_max_token_length
                               , hparam.pre_seq_len
                               , hparam.prefix_max_token_length)


train_dataset = load_dataset('')
val_dataset = load_dataset('')
init_text_datasets(train_dataset, val_dataset)

train_loader = DataLoader(train_dataset, batch_size=hparam.batch_size, collate_fn=collator, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hparam.batch_size, collate_fn=collator)

train_length = len(train_loader)
for epoch in range(hparam.num_epoch):
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()        
        # Forward pass
        outputs = model(st_inputs = batch['st_inputs']
                        , prefix_inputs = batch['prefix_inputs']
                        , interview_lengths = batch['interview_lengths'])
        
        # Compute the loss
        loss = loss_function(outputs.logits, batch['labels'].to('cuda'))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(st_inputs = batch['st_inputs']
                        , prefix_inputs = batch['prefix_inputs']
                        , interview_lengths = batch['interview_lengths'])
        
            loss = loss_function(outputs.logits, batch['labels'].to('cuda'))
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_loader)

    # Print or log the losses
    print(f'Epoch {epoch+1}/{hparam.num_epoch}, Loss: {average_loss}, Val Loss: {average_val_loss}')