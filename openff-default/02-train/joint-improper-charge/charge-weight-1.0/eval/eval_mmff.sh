#!/bin/bash
#BSUB -P "esp"
#BSUB -J "eval-[1-150]"
#BSUB -n 1
#BSUB -R rusage[mem=384]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 2:00
#BSUB -o out/out_%J_%I.stdout
#BSUB -eo out/out_%J_%I.stderr
#BSUB -L /bin/bash

# parameters
forcefield='mmff94'
layer="SAGEConv"
units=512
activation="relu"
config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
input_prefix="../datasets/${forcefield}"
exp='blooming-salad-32'
epoch=500
checkpoint_path="../checkpoints_mmff_rdkit/$exp"

# temporal directory
mkdir -p pkl out

# run
datasets="gen2 gen2-torsion pepconf-dlc protein-torsion spice-pubchem spice-dipeptide spice-des-monomers"



python eval_mmff.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" \
       --input_prefix $input_prefix --datasets="$datasets" --checkpoint_path $checkpoint_path -e $epoch 

