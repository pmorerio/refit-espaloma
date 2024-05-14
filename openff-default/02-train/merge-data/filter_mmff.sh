#!/bin/bash
#BSUB -P "filter"
#BSUB -J "filter"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q cpuqueue
#BSUB -sp 1 # low priority. default is 12, max is 25
#BSUB -W 24:00
#BSUB -o stdout/filter_%J_%I.stdout
#BSUB -eo stderr/filter_%J_%I.stderr
#BSUB -L /bin/bash


OPENMM_CPU_THREADS=1

base_forcefield='mmff94'

datasets=$(ls ${base_forcefield})


for dataset in $datasets
do
    echo $dataset
    mkdir -p ${base_forcefield}_filtered/${dataset}
    python ./script/filter_mmff.py --dataset ${dataset} --base_forcefield $base_forcefield
done
