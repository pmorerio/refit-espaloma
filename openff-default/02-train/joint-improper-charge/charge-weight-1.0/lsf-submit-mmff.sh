#!/bin/bash
#BSUB -P "esp"
#BSUB -J "esp"
#BSUB -n 1
#BSUB -R rusage[mem=256]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -gpu num=1:j_exclusive=yes:mode=shared
#BSUB -W 35:59
#BSUB -o out_%J_%I.stdout
#BSUB -eo out_%J_%I.stderr
#BSUB -L /bin/bash

# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# settings
epochs=1500
batch_size=64
layer="SAGEConv"
units=512
activation="relu"
config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
learning_rate=1e-4
input_prefix="datasets/mmff94"
datasets="gen2 gen2-torsion pepconf-dlc protein-torsion spice-pubchem spice-dipeptide spice-des-monomers"


exp='blooming-salad-32'
output_prefix="checkpoints_mmff_rdkit/$exp"


n_max_confs=50
force_weight=1.0
load=1

# run
python train_mmff.py --epochs $epochs --batch_size $batch_size --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" --learning_rate $learning_rate --input_prefix $input_prefix --datasets "$datasets" --output_prefix $output_prefix --n_max_confs $n_max_confs --force_weight $force_weight --load $load
