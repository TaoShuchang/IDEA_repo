# Poisoning attack (MetaAttack) at training time
# Cora
python -u main.py --dataset cora --lambda_inv_risks 100 --num_sample 2 --perturb_size 1e-4

# Citeseer
python -u main.py --dataset citeseer  --lambda_inv_risks 50 --perturb_size 1e-4

# Reddit
python -u main.py --dataset reddit --lambda_inv_risks 100 --suffix idea --num_sample 2 --perturb_size 1e-4

# ogbn-products
python -u main.py --dataset ogbproducts --num_sample 4 

# ogbn-arxiv
nohup python -u main.py --dataset ogbarxiv --num_sample 2


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
