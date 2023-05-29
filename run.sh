# Poisoning attack (MetaAttack) at training time
# Cora
python -u main.py --dataset cora --alpha 100  --dom_num 10 

# Citeseer
python -u main.py --dataset citeseer  --alpha 100 --dom_num 10  --num_sample 8

# Reddit
python -u main.py --dataset reddit --alpha 100 --dom_num 10 --num_sample 4 --perturb_size 1e-3

# ogbn-products
python -u main.py --dataset ogbproducts --alpha 10 --dom_num 10 --num_sample 4 --perturb_size 1e-3

# ogbn-arxiv
nohup python -u main.py --dataset ogbarxiv --alpha 10 --dom_num 10 --perturb_size 1e-3


# Evasion attack  (nettack, PGD, G-NIA, TDGIA) at testing time
# Cora
python -u test.py --dataset cora 

# Citeseer
python -u test.py --dataset citeseer 

# Reddit
python -u test.py --dataset reddit 

# ogbn-products
python -u test.py --dataset ogbproducts 

# ogbn-arxiv
nohup python -u test.py --dataset ogbarxiv
