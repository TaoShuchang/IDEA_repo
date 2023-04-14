#!/bin/bash  
#This is the basic example of "for loop".  

defense=("GCN" "GAT" "APPNP" "RGCN" "GNNGuard" "PROGNN" "SimpGCN" "Elastic" "STABLE")
namearr=("log" "checkpoint")
dataarr=("12k_reddit" "10k_ogbproducts" "ogbarxiv" "citeseer" "cora" "pubmed")



for def in ${defense[@]}
do
for name in ${namearr[@]}
do
for dataset in ${dataarr[@]}
do  
str=$def"/"$name"/"$dataset"/"
echo $str
mkdir -p $str
done  
done
done

# atkarr=("TDGIA" "PGD" "GNIA")
# name="new_graphs"
# for atk in ${atkarr[@]}
# do
# for dataset in ${dataarr[@]}
# do  
# mkdir -p "final_graphs/"$dataset"/"
# str=$atk"/"$name"/"$dataset"/"
# echo $str
# cp $str*".npz" "final_graphs/"$dataset"/"
# done
# done


echo "Thank You."


