#!/usr/bin/env python

import os, sys
import numpy as np
import h5py
import torch
import espaloma as esp
from espaloma.units import *
import click
from openff.toolkit.topology import Molecule
from simtk import unit
from simtk.unit import Quantity



def get_graph(record, key):
    # set default to float64
    #torch.set_default_dtype(torch.float64)
    #print("torch default dtype is now {}".format(torch.get_default_dtype()))

    smi = record["smiles"][0].decode('UTF-8')
    offmol = Molecule.from_mapped_smiles(smi, allow_undefined_stereo=True)
    #offmol.compute_partial_charges_am1bcc()   # https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Molecule.html
    try:
        offmol.assign_partial_charges(partial_charge_method="am1bccelf10")
    except:
        failures = open('../../partial_charge_failures.txt', 'a')
        failures.write("{}\t{}\n".format(key, smi))
        failures.close()
        msg = 'could not assign partial charge'
        raise ValueError(msg)
    charges = offmol.partial_charges.value_in_unit(esp.units.CHARGE_UNIT)
    g = esp.Graph(offmol)

    # energy
    energy = []
    for e, e_corr in zip(record["dft_total_energy"], record["dispersion_correction_energy"]):
        energy.append(e + e_corr)
    
    # gradient
    grad = []
    for gr, gr_corr in zip(record["dft_total_gradient"], record["dispersion_correction_gradient"]):
        grad.append(gr + gr_corr)
        
    # conformations
    conformations = record["conformations"]

    # energy is already hartree
    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                _energy,
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
            for _energy in energy
        ],
        #dtype=torch.get_default_dtype(),
        dtype=torch.float64,
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    xyz,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
                for xyz in conformations
            ],
            axis=1,
        ),
        requires_grad=True,
        #dtype=torch.get_default_dtype(),
        dtype=torch.float32,
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.stack(
        [
            torch.tensor(
                Quantity(
                    _grad,
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                #dtype=torch.get_default_dtype(),
                dtype=torch.float32,
            )
            for _grad in grad
        ],
        dim=1,
    )
    
    #g.nodes['n1'].data['q_ref'] = c = torch.tensor(charges, dtype=torch.get_default_dtype(),).unsqueeze(-1)
    g.nodes['n1'].data['q_ref'] = c = torch.tensor(charges, dtype=torch.float32,).unsqueeze(-1)
    
    return g



def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    key = kwargs["keyname"]
    output_prefix = kwargs["output_prefix"]

    hdf = h5py.File(filename)
    record = hdf[key]

    g = get_graph(record, key)
    g.save(output_prefix)



@click.command()
@click.option("--hdf5", required=True, help='hdf5 filename')
@click.option("--keyname", required=True, help='keyname of the hdf5 group')
@click.option("--output_prefix", required=True, help='output directory to save graph data')
def cli(**kwargs):
    print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()
