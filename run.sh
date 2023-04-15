## cora
python -u main.py --dataset cora --lambda_inv_risks 100 --num_sample 2 --perturb_size 1e-4

## citeseer
python -u main.py --dataset citeseer  --lambda_inv_risks 50 --perturb_size 1e-4

## reddit
python -u main.py --dataset reddit --lambda_inv_risks 100 --suffix idea --num_sample 2 --perturb_size 1e-4

## ogbproducts
python -u main.py --dataset ogbproducts --num_sample 4 

## ogbarxiv
nohup python -u main.py --dataset ogbarxiv --num_sample 2
