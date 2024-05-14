# imports
import os

import espaloma as esp
import torch
# define or load a molecule of interest via the Open Force Field toolkit
from openff.toolkit.topology import Molecule


class GetLoss(torch.nn.Module):
    def forward(self, g):
        return torch.nn.MSELoss()(
            g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
            g.nodes['g'].data['u_ref_relative'],
        )

_ds_vl = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
molecule = Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
# create an Espaloma Graph object to represent the molecule of interest
molecule_graph = esp.Graph(molecule)

# load local pretrained model
espaloma_model = torch.load("../deployed_models/net1500.pt")
espaloma_model.eval()

# apply a trained espaloma model to assign parameters
out = espaloma_model(molecule_graph.heterograph)

# create an OpenMM System for the specified molecule
openmm_system = esp.graphs.deploy.openmm_system_from_graph(molecule_graph)
breakpoint()
