#!/bin/bash
#BSUB -P "merge"
#BSUB -J "pubchem"
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
forcefields="gaff-1.81 gaff-2.11 openff-1.2.0 openff-2.0.0"

path_to_dataset="../dataset"
datasets="gen2 spice-des-monomers spice-dipeptide spice-pubchem gen2-torsion pepconf-dlc protein-torsion"

for sub in $datasets; do
    python ./script/calc_ff.py --path_to_dataset $path_to_dataset --dataset $sub --forcefields "$forcefields" --base_forcefield 'openff-2.0.0'
done
