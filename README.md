# Dual Encoder Model
This repo contains code related to the [dual encoder model](https://www.frontiersin.org/articles/10.3389/fpsyt.2023.1160291/full) from our work in "Automatic Depression Severity Assessment with Deep Learning Using Parameter-Efficient Tuning", which has been published at Frontiers in Psychiatry, section Digital Mental Health.

## Data
The [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu) is used for this experiement. The data can be obtained after signing an agreement form. This dataset contains interviews with 189 subjects. Raw audio files, transcriptions, and facial landmark features are provided. 

## Files
    .
    ├── config                      # Code for model config
    │   └── dual_encoder.yaml       # hparam config for dual encoder
    │
    ├── dataset                     # Code for preprocessing DAIC-WOZ
    │   └── dataset.py              # Custom Dataset class code.
    │   └── preprocessing.py        # DAIC-WOZ transcript preprocessing code.
    │
    ├── model                       # Code for initializing a dual encoder model
    │   ├── attention.py            # Implementation of attention layer for BiLSTM.
    │   ├── dual_encoder.py         # Dual encoder model code.
    │   ├── layer_init.py           # Layer initialization code.
    │   ├── prefix_encoder.py       # Prefix encoder-only model code.
    │   └── utils.py                # Training setup code.
    └── ...

## Reference
For more details of the methods and results, please refer to our paper.
```
@article{lau2023automatic,
  title={Automatic depression severity assessment with deep learning using parameter-efficient tuning},
  author={Lau, Clinton and Zhu, Xiaodan and Chan, Wai-Yip},
  journal={Frontiers in Psychiatry},
  volume={14},
  pages={1160291},
  year={2023},
  publisher={Frontiers}
}
```

