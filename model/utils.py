import os
import numpy as np
import torch
import random
import yaml
from attrdict import AttrDict
from os.path import join
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, mean_absolute_error

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_classification_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def compute_regression_metrics(p):
    pred, labels = p
    
    rmse = mean_squared_error(labels, pred, squared=False)
    mae = mean_absolute_error(labels, pred)

    return {"rmse": rmse, "mae": mae}

def config_run(config_path):
    cfg = load_config_yaml(config_path)
    hparam = cfg.hparam
    device = setup_gpu(hparam.device_id)
    set_seed(hparam.seed)

    return cfg, hparam, device

def setup_gpu(device_id):
    GPU_NUM = ','.join(map(str, device_id))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return device

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config_yaml(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = AttrDict(config)
    
    return cfg