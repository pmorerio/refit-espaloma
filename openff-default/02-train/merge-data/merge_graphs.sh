#!/bin/bash
#BSUB -P "merge"
#BSUB -J "merge"
#BSUB -n 1
#BSUB -R rusage[mem=16]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 24:00
#BSUB -o stdout/merge_%J_%I.stdout
#BSUB -eo stderr/merge_%J_%I.stderr
#BSUB -L /bin/bash

BASE_FORCEFIELD='mmff94'

python ./script/merge_graphs.py --base_forcefield $BASE_FORCEFIELD
