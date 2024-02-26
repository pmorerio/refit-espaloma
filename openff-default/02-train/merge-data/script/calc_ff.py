#!/usr/bin/env python
import copy
import glob
import os
import sys

import click
import espaloma as esp
import numpy as np
import torch
from espaloma.data.md import *
from openff.toolkit.topology import Molecule
from openmm import openmm, unit
from openmm.app import Simulation
from openmm.unit import Quantity
from openmmforcefields.generators import SystemGenerator
from tqdm import tqdm

# Basic settings


# Simulation Specs
TEMPERATURE = 350 * unit.kelvin
STEP_SIZE = 1.0 * unit.femtosecond
COLLISION_RATE = 1.0 / unit.picosecond
EPSILON_MIN = 0.05 * unit.kilojoules_per_mole



def baseline_energy_force(g, forcefield):
    """
    Calculate baseline energy using legacy forcefields
    
    reference:
    https://github.com/choderalab/espaloma/espaloma/data/md.py
    """

    if forcefield in ['gaff-1.81', 'gaff-2.11', 'openff-1.2.0', 'openff-2.0.0']:
        generator = SystemGenerator(
            small_molecule_forcefield=forcefield,
            molecules=[g.mol],
            forcefield_kwargs={"constraints": None, "removeCMMotion": False},
        )
        suffix = forcefield
    elif forcefield in ['amber14-all.xml']:
        generator = SystemGenerator(
            forcefields=[forcefield],
            molecules=[g.mol],
            forcefield_kwargs={"constraints": None, "removeCMMotion": False},
        )
        suffix = "amber14"
    else:
        raise Exception('force field not supported')


    # parameterize topology
    topology = g.mol.to_topology().to_openmm()

    # create openmm system
    system = generator.create_system(
        topology,
    )

    # use langevin integrator, although it's not super useful here
    integrator = openmm.LangevinIntegrator(
        TEMPERATURE, COLLISION_RATE, STEP_SIZE
    )

    # create simulation
    simulation = Simulation(
        topology=topology, system=system, integrator=integrator
    )

    # get energy
    us = []
    us_prime = []
    xs = (
        Quantity(
            g.nodes["n1"].data["xyz"].detach().numpy(),
            esp.units.DISTANCE_UNIT,
        )
        .value_in_unit(unit.nanometer)
        .transpose((1, 0, 2))
    )
    for x in xs:
        simulation.context.setPositions(x)
        us.append(
            simulation.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(esp.units.ENERGY_UNIT)
        )
        us_prime.append(
            simulation.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(esp.units.FORCE_UNIT) * -1
        )

    #us = torch.tensor(us)[None, :]
    us = torch.tensor(us, dtype=torch.float64)[None, :]
    us_prime = torch.tensor(
        np.stack(us_prime, axis=1),
        dtype=torch.get_default_dtype(),
    )

    g.nodes['g'].data['u_%s' % suffix] = us
    g.nodes['n1'].data['u_%s_prime' % suffix] = us_prime

    return g



def run(kwargs):
    path_to_dataset = kwargs['path_to_dataset']
    dataset = kwargs['dataset']
    forcefields = kwargs['forcefields']
    base_forcefield = kwargs['base_forcefield']

    # convert forcefields into list
    _forcefields = []
    for _ in forcefields.split():
        try:
            _forcefields.append(int(_))
        except:
            _forcefields.append(str(_))
    forcefields= _forcefields


    entry_path = os.path.join(path_to_dataset, dataset)
    paths_to_mydata = glob.glob("{}/*/mydata".format(entry_path))


    n_total_confs = 0
    n_total_mols = len(paths_to_mydata)


    with open("calc_ff_{}.log".format(dataset), "w") as wf:
        wf.write(">{}: {} molecules found\n".format(dataset, n_total_mols))
        for p in tqdm(paths_to_mydata):
            try:
                entry_id = p.split('/')[-2]  # str
                save_path = '{}/{}/{}'.format(base_forcefield, dataset, entry_id)
                if os.path.exists(save_path):
                    continue
                
                _g = esp.Graph.load(p)   # graph generated from hdf5 file. u_ref corresponds to native QM energy.
                g = copy.deepcopy(_g)
                n_confs = g.nodes['n1'].data['xyz'].shape[1]
                n_total_confs += n_confs

                """
                subtract nonbonded from qm and calculate legacy forcefields
                """
                
                # clone qm energy
                g.nodes['g'].data['u_qm'] = g.nodes['g'].data['u_ref'].detach().clone()
                g.nodes['n1'].data['u_qm_prime'] = g.nodes['n1'].data['u_ref_prime'].detach().clone()

                
                # subtract nonbonded interaction energy. u_ref will be overwritten.
                g = subtract_nonbonded_force(g, forcefield=base_forcefield, subtract_charges=True)

                # calculate baseline energy with legacy forcefields # TODO (gianscarpe): in mmff94 as well?
                for forcefield in forcefields:
                    g = baseline_energy_force(g, forcefield)

                """
                report
                """
                
                wf.write("{:8d}: {:4d} conformations found\n".format(int(entry_id), n_confs))


                """
                save graph
                """
                g.save(save_path)
            except:
                _g = esp.Graph.load(p)   # graph generated from hdf5 file. u_ref corresponds to native QM energy.
                print(_g.mol.to_smiles())

        # summary
        wf.write("------------------\n")
        wf.write(">total molecules: {}\n".format(n_total_mols))
        wf.write(">total conformations: {}\n".format(n_total_confs))
        


@click.command()
@click.option("--path_to_dataset", required=True, help="path to the dataset")
@click.option("--dataset",  required=True, type=click.Choice(['gen2', 'gen2-torsion', 'pepconf', 'pepconf-dlc', 'protein-torsion', 'rna-diverse', 'rna-trinucleotide', 'rna-nucleoside', 'spice-dipeptide', 'spice-pubchem', 'spice-des-monomers']), help="name of the dataset")
@click.option("--forcefields", required=True, help="legacy forcefields in sequence [gaff-1.81, gaff-2.10, openff-1.2.0, openff-2.0.0, amber14-all.xml]", type=str)
@click.option("--base_forcefield", required=True, help="base forcefield in [mmff94, openff-1.2.0, openff-2.0.0]", default='openff-2.0.0', type=str)
def cli(**kwargs):
    print(kwargs)
    print(esp.__version__)
    run(kwargs)



if __name__ == '__main__':
    cli()
