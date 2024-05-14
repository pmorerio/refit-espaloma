#!/usr/bin/env python
import glob
import math
import os
import random
import re
import subprocess
import sys

import click
import dgl
import espaloma as esp
import numpy as np
import torch
import tqdm

# Settings
HARTEE_TO_KCALPERMOL = 627.509
BOHR_TO_ANGSTROMS = 0.529177
RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

class GetLoss(torch.nn.Module):
    def energy_loss(self, g):
        return torch.nn.MSELoss()(
            g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
            g.nodes['g'].data['u_ref_relative'],
        )
    def force_loss(self, g):
        du_dx_hat = torch.autograd.grad(
            g.nodes['g'].data['u'].sum(),
            g.nodes['n1'].data['xyz'],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        du_dx = g.nodes["n1"].data["u_ref_prime"].float()

        return torch.nn.MSELoss()(
            du_dx, 
            du_dx_hat
        )
    def forward(self, g):
        return self.energy_loss(g), self.force_loss(g)


def _compute_lin_ligen(g):
    """
    Remove unnecessary data from graph.
    """

    LIGEN_N_COLUMNS = 12
    LIGEN_LIN_MAPPING = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    mol = g.mol.to_smiles()
    result = subprocess.run('ligen-type', input=mol, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # parsing ligen output for lin
    idxs = [int(re.findall("\d+", a[a.find("#"):])[0]) for a in result.stdout.split("\n") if len(a.split()) >= LIGEN_N_COLUMNS]

    if idxs:
        
        mol_lins_mapping = [LIGEN_LIN_MAPPING[idx] for idx in idxs]
        lins = [mol_lins_mapping[i1] for _, i1, _ in g.nodes['n3'].data['idxs']]

        
        g.nodes['n3'].data['lin'] = torch.tensor(lins)[:, None].bool()
        return g
    else:
        raise Exception(f"Cannot handle molecule {mol}")

def _compute_lin(g):
    """
    Remove unnecessary data from graph.
    """


    LIGEN_LIN_MAPPING = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0])

    # make sure mol is in openff.toolkit format `
    mol = g.mol

    import rdkit.Chem.rdForceFieldHelpers as ff
    from rdkit.Chem import AllChem
    mol = mol.to_rdkit()
    mmff = ff.MMFFGetMoleculeProperties(mol)

    idxs = [mmff.GetMMFFAtomType(i) for i in range(mol.GetNumAtoms())]

    if idxs:
        mol_lins_mapping = [LIGEN_LIN_MAPPING[idx-1] for idx in idxs]
        lins = [mol_lins_mapping[i1] for _, i1, _ in g.nodes['n3'].data['idxs']]
        g.nodes['n3'].data['lin'] = torch.tensor(lins)[:, None].bool()
        return g
    else:
        raise Exception(f"Cannot handle molecule {mol}")# put types into graph object


def run(kwargs):
    layer = kwargs['layer']
    units = kwargs['units']
    config = kwargs['config']
    janossy_config = kwargs['janossy_config']
    datasets = kwargs['datasets']
    input_prefix = kwargs['input_prefix']
    checkpoint_path = kwargs['checkpoint_path']
    epoch = kwargs['epoch']
    validation=True

    # Convert config and janossy_config into list
    _config = []
    for _ in config.split():
        try:
            _config.append(int(_))
        except:
            _config.append(str(_))
    config = _config

    _janossy_config = []
    for _ in janossy_config.split():
        try:
            _janossy_config.append(int(_))
        except:
            _janossy_config.append(str(_))
    janossy_config = _janossy_config


    #
    # Espaloma model
    #
    layer = esp.nn.layers.dgl_legacy.gn(layer, {"aggregator_type": "mean", "feat_drop": 0.1})
    representation = esp.nn.Sequential(layer, config=config)

    # out_features: Define modular MM parameters Espaloma will assign
    # 1: atom hardness and electronegativity
    # 2: bond linear combination, enforce positive
    # 3: angle linear combination, enforce positive
    # 4: torsion barrier heights (can be positive or negative)
    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=units, config=janossy_config,
        out_features={
                1: {'s': 1, 'e': 1},
                2: {'eq':1, 'k': 1}, # 1?
                3: {'k': 1, 'eq':1,  'kstretch': 2},
                4: {'k': 3},
        },
    )
    readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=janossy_config, out_features={"k": 2})

    readout_oop = esp.nn.readout.janossy.JanossyPoolingOOP(in_features=units, config=janossy_config, out_features={"k": 1})

    energy_graph = esp.mm.energy.EnergyInGraphMMFF(terms=["n2", "n3", "n4", "n4_improper", "n4_oop"])
    
    net = torch.nn.Sequential(
        representation,
        readout,
        readout_improper,
        readout_oop,
        #esp.nn.readout.janossy.ExpCoefficients(),
        esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
        esp.mm.geometry.GeometryInGraph(),
        energy_graph,
        GetLoss(),
    )

    #
    # Calculate rmse loss
    #

    device = 'cpu'
    print(f"Loading checkpoint at {checkpoint_path}")
    state_dict = torch.load(os.path.join(checkpoint_path, "net{}.th".format(epoch)), map_location=torch.device('cpu'))


    net.load_state_dict(state_dict)
    net.eval()
    
    print("saving {}".format(epoch))
    torch.save(net, f'net{epoch}' + ".pt")


    # Convert datasets into list
    _datasets = [ str(_) for _ in datasets.split() ]
    datasets = _datasets

    net = net.to(device)
    #
    # Load datasets
    #
    print("# LOAD UNIQUE MOLECULES")
    for i, dataset in enumerate(datasets):
        path = os.path.join(input_prefix, dataset)
        ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        _ds_tr, _ds_vl, _ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

        # Merge datasets
        if i == 0:
            ds_vl = _ds_vl
            ds_te = _ds_te
        else:
            ds_vl += _ds_vl
            ds_te += _ds_te
        del ds, _ds_tr, _ds_vl, _ds_te

    #
    # Load duplicated molecules
    #
    print("# LOAD DUPLICATED MOLECULES")
    entries = glob.glob(os.path.join(input_prefix, "duplicated-isomeric-smiles-merge", "*"))
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    n_entries = len(entries)
    entries_tr = entries[:int(n_entries*TRAIN_RATIO)]
    entries_vl = entries[int(n_entries*TRAIN_RATIO):int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO)]
    entries_te = entries[int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO):]
    print("Found {} entries. Split data into {}:{}:{} entries.".format(n_entries, len(entries_tr), len(entries_vl), len(entries_te)))
    assert n_entries == len(entries_tr) + len(entries_vl) + len(entries_te)


    print(f"The final validate and test data size is {len(ds_vl)} and {len(ds_te)}.")

    #
    # Remove unnecessary data from graph
    #
    from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers

    def fn(g):
        # remove
        g.nodes['g'].data.pop('u_qm')
        g.nodes['g'].data.pop('u_gaff-1.81')
        g.nodes['g'].data.pop('u_gaff-2.11')
        g.nodes['g'].data.pop('u_openff-1.2.0')
        g.nodes['g'].data.pop('u_openff-2.0.0')
        try:
            g.nodes['g'].data.pop('u_amber14')
        except:
            pass
        g.nodes['g'].data.pop('u_ref')
        g.nodes['n1'].data.pop('q_ref')        
        g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref_relative'].float()
        return g


    def _fn(g):
        """
        Remove unnecessary data from graph.
        """
        g.nodes['g'].data.pop('u_qm')
        g.nodes['g'].data.pop('sum_q') # added (gianscarpe)
        g.nodes['g'].data.pop('u_gaff-1.81')
        g.nodes['g'].data.pop('u_gaff-2.11')
        g.nodes['g'].data.pop('u_openff-1.2.0')
        g.nodes['g'].data.pop('u_openff-2.0.0')
        g.nodes['n1'].data.pop('u_qm_prime')
        g.nodes['n1'].data.pop('u_gaff-1.81_prime')
        g.nodes['n1'].data.pop('u_gaff-2.11_prime')
        g.nodes['n1'].data.pop('u_openff-1.2.0_prime')
        g.nodes['n1'].data.pop('u_openff-2.0.0_prime')
        try:
            g.nodes['g'].data.pop('u_amber14')
            g.nodes['n1'].data.pop('u_amber14_prime')
        except:
            pass
        # Remove u_ref_relative. u_ref_relative will be recalculated after handling heterographs with different size
        
        g.nodes['g'].data['u_ref'] = g.nodes['g'].data['u_ref'].double()
        g.nodes['n1'].data['q_ref'] = g.nodes['n1'].data['q_ref'].float()

        return g


    from espaloma.graphs.utils.generate_oops import generate_oops


    if validation:
        for entry in entries_vl:
            _datasets = os.listdir(entry)
            for _dataset in _datasets:
                _ds_vl = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                ds_vl += _ds_vl
                del _ds_vl

        ds_vl.apply(_fn, in_place=True)
        ds_vl.apply(_compute_lin, in_place=True)
        ds_vl.apply(generate_oops, in_place=True)
        ds_vl.apply(regenerate_impropers, in_place=True)
        
        
        

    else:
        for entry in entries_te:
            _datasets = os.listdir(entry)
            for _dataset in _datasets:
                _ds_te = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
                ds_te += _ds_te
                del _ds_te

        ds_te.apply(fn, in_place=True)
        ds_te.apply(_compute_lin, in_place=True)
        ds_te.apply(generate_oops, in_place=True)
        ds_te.apply(regenerate_impropers, in_place=True)
        


    # Calc

    e_vl, e_te = [], []
    f_vl, f_te = [], []
    myresults = {}
    # Validation
    
    if validation:
        results = []
        print("validation")
        for g in tqdm.tqdm(ds_vl):
                # apply network and get energy loss
                g.nodes["n1"].data["xyz"].requires_grad = True
                het = g.heterograph.to(device)
                e_loss, f_loss = net(het)
                e = HARTEE_TO_KCALPERMOL * e_loss.pow(0.5).item()
                e_vl.append(e)

                f = (HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS) * f_loss.pow(0.5).item()
                f_vl.append(f)

                results.append((g.mol.to_smiles(), e, f, g.nodes['n3'].data['lin'].bool().any(), g.path.split("/")[3],
                                het.nodes['g'].data['u_n4_oop'].mean().detach()  if 'u_n4_oop' in het.nodes['g'] else None,
                                het.nodes['g'].data['u_n4'].mean().detach()  if 'u_n4' in het.nodes['g'] else None,
                                het.nodes['g'].data['u_n4_improper'].mean().detach()  if 'u_n4_improper' in het.nodes['g'] else None,
                                het.nodes['g'].data['u_n2'].mean().detach(),
                                het.nodes['g'].data['u_n3_sb'].mean().detach(),
                                het.nodes['g'].data['u'].mean().detach(),
                                het.nodes['g'].data['u_n3'].mean().detach(),
                                ))

        import pickle
        with open('./classii.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        e_vl = np.array(e_vl).mean()
        f_vl = np.array(f_vl).mean()
        myresults['e_vl'] = e_vl
        myresults["f_vl"] = f_vl
        print(myresults)
        
        with open('./pkl/{}_val.pickle'.format(epoch), 'wb') as handle:
            pickle.dump(myresults, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    else:
        del ds_vl
    # Test
    print("testing")
    
    for g in tqdm.tqdm(ds_te):
        # apply network and get energy loss
        g.nodes["n1"].data["xyz"].requires_grad = True
        e_loss, f_loss = net(g.heterograph.to(device))  # energy
        e_te.append(HARTEE_TO_KCALPERMOL * e_loss.pow(0.5).item())
        f_te.append((HARTEE_TO_KCALPERMOL/BOHR_TO_ANGSTROMS) * f_loss.pow(0.5).item())

    # Energy and Force

    e_te = np.array(e_te).mean()
    
    f_te = np.array(f_te).mean()
    myresults["e_te"] = e_te
    myresults["f_te"] = f_te
    print(myresults)
    # Save
    import pickle
    with open('./pkl/{}.pickle'.format(epoch), 'wb') as handle:
        pickle.dump(myresults, handle, protocol=pickle.HIGHEST_PROTOCOL)



@click.command()
@click.option("-l",   "--layer",           default="SAGEConv", type=click.Choice(["SAGEConv", "GATConv", "TAGConv", "GINConv", "GraphConv"]), help="GNN architecture")
@click.option("-u",   "--units",           default=128, help="GNN layer", type=int)
@click.option("-act", "--activation",      default="relu", type=click.Choice(["relu", "leaky_relu"]), help="activation method")
@click.option("-c",   "--config",          default="128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-jc",  "--janossy_config", default="128 relu 128 relu 128 relu 128 relu", help="sequence of numbers (for units) and strings (for activation functions)", type=str)
@click.option("-i",   "--input_prefix",    default="data", help="input prefix to graph data", type=str)
@click.option("-d",   "--datasets",        help="name of the datasets", type=str)
@click.option("-o",   "--checkpoint_path", default="../checkpoints", help="path to checkpoint network models", type=str)
@click.option("-e",   "--epoch",           required=True, help="epoch number", type=int)
def cli(**kwargs):
    epoch = kwargs['epoch']
    if os.path.exists(f"./pkl/{epoch}.pickle") and os.path.exists(f"./pkl/{epoch}_val.pickle"):
        exit()
    else:
        print(kwargs)
        run(kwargs)



if __name__ == "__main__":
    cli()

