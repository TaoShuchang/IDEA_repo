# IDEA

**This repository is our Pytorch implementation of our paper:**

**[IDEA: Invariant Causal Learning for Graph Adversarial Robustness against Attacks](https://openreview.net/pdf?id=mQy_8YPulG)**
Submitted to KDD 2023  [openreview](https://openreview.net/forum?id=mQy_8YPulG)

## Datasets and splits

- Download  Cora, Citeseer, ogbarxiv, ogbproducts (the subgraph in our paper), Reddit (the subgraph in our paper) from this [Here](https://drive.google.com/file/d/1xD2GReVHaN7zUTSfM8lXDie9NtpiyP0g/view?usp=share_link).

- Dowload the train/val/test splits mentioned in our paper are also include in the above link [Here](https://drive.google.com/file/d/1xD2GReVHaN7zUTSfM8lXDie9NtpiyP0g/view?usp=share_link). Please note that, for Cora and Citeseer, we adopt the commonly used splits which are included in `datasets/Cora/` and `datasets/Citeseer/`.

Unzip the  `idea_Data_Split.zip`, and put the two folders (`datasets` and `splits`) in this directory.



## Environment

- python >= 3.9
- pytorch == 1.11.0--cuda11.3
- scipy == 1.9.3
- numpy == 1.23.5
- deeprobust
- ogb




## Attacked graphs 

Download the attacked graphs used in our paper from [Here](https://drive.google.com/file/d/1fcpPBgSF_Kkexx4egSN8ShqArWByh1Fl/view?usp=share_link). 

Unzip `attacked_graph.zip`, and put the folder `attacked_graphs` in this directory.



## Reproduce the results

We provide to evaluate the robustness of IDEA by poisoning and evasion attacks.

- Poisoning attack (MetaAttack in our paper)

  *Example Usage*

  ```python
  python -u main.py --dataset cora --lambda_inv_risks 100 --num_sample 2 --perturb_size 1e-4 
  ```


- Evasion attack (nettack, PGD, G-NIA, TDGIA in our paper)

  *Example Usage*

  ```python
  python -u test.py --dataset cora
  ```
  



**Running scripts and parameters for all the datasets are given in `run.sh`**


