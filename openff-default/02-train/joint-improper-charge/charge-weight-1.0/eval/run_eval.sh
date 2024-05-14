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

OPENMM_CPU_THREADS=1
#export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env

# Report node in use
echo "======================"
hostname
env | sort | grep 'CUDA'
nvidia-smi
echo "======================"


# parameters
layer="SAGEConv"
units=512
activation="relu"
config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
janossy_config="${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1 ${units} relu 0.1"
input_prefix="../merge-data"
checkpoint_path="../checkpoints"
epoch=1500
# temporal directory
mkdir -p pkl out

# conda
# run
datasets="gen2 spice-pubchem"
python eval.py --layer $layer --units $units --activation $activation --config "$config" --janossy_config "$janossy_config" \
--input_prefix $input_prefix --datasets="$datasets" --checkpoint_path $checkpoint_path --epoch $(( $epoch ))

