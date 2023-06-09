# Poisoning attack (MetaAttack) at training time
# Cora
nohup python -u main.py --dataset cora --alpha 100  --dom_num 10  --device 1 > log/cora.log 2>&1 &

# Citeseer
nohup python -u main.py --dataset citeseer  --alpha 10 --dom_num 10 --perturb_size 1e-4 --num_sample 8 > log/citeseer.log 2>&1 &

# Reddit
nohup python -u main.py --dataset reddit --alpha 10 --dom_num 10 --device 2 > log/reddit.log 2>&1 &

# ogbn-products
nohup python -u main.py --dataset ogbproducts --alpha 10 --dom_num 10 --device 3 > log/ogbproducts.log 2>&1 &

# ogbn-arxiv
nohup nohup python -u main.py --dataset ogbarxiv --alpha 10 --dom_num 10 > log/ogbarxiv.log 2>&1 &


# Evasion attack  (nettack, PGD, TDGIA, and G-NIA) at testing time
# Cora
nohup python -u test.py --dataset cora > log/eva_cora.log 2>&1 &

# Citeseer
nohup python -u test.py --dataset citeseer  > log/eva_citeseer.log 2>&1 &

# Reddit
nohup python -u test.py --dataset reddit  > log/eva_reddit.log 2>&1 &

# ogbn-products
nohup python -u test.py --dataset ogbproducts  > log/eva_ogbproducts.log 2>&1 &

# ogbn-arxiv
nohup python -u test.py --dataset ogbarxiv  > log/eva_ogbarxiv.log 2>&1 &
