#!/bin/bash
#BSUB -P "merge"
#BSUB -J "gen2"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 47:00
#BSUB -o stdout/calc_%J_%I.stdout
#BSUB -eo stderr/calc_%J_%I.stderr
#BSUB -L /bin/bash

OPENMM_CPU_THREADS=1
export OE_LICENSE=~/.openeye/oe_license.txt   # Open eye license activation/env

# run job
dataset="gen2"
mkdir -p openff-2.0.0/${dataset}
forcefields="gaff-1.81 gaff-2.11 openff-1.2.0 openff-2.0.0"
path_to_dataset="../dataset"

python ./script/calc_ff.py --path_to_dataset $path_to_dataset --dataset $dataset --forcefields "$forcefields"
