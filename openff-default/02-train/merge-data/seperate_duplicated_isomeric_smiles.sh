#!/bin/bash
#BSUB -P "duplicate"
#BSUB -J "duplicate"
#BSUB -n 1
#BSUB -R rusage[mem=256]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 24:00
#BSUB -o stdout/duplicated_%J_%I.stdout
#BSUB -eo stderr/duplicated_%J_%I.stderr
#BSUB -L /bin/bash


BASE_FORCEFIELD='mmff94'

# run job
python ./script/seperate_duplicated_isomeric_smiles.py --base_forcefield $BASE_FORCEFIELD
