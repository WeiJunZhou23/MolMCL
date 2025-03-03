import os
import pickle
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from molmcl.utils.data import *

class MoleculeDataset_cm(Dataset):
    def __init__(self, data_smiles, data_labels, feat_type):
        self.feat_type = feat_type
        smiles = data_smiles
        labels = data_labels
        #labels = labels.replace(0, -1)
        labels = labels.values
        
        # convert mol to graph with smiles validity filtering
        self.smiles, self.labels, self.mol_data = [], [], []
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                data = eval('mol_to_graph_data_obj_{}'.format(feat_type))(mol)
                self.smiles.append(smi)
                self.labels.append(labels[i])
                self.mol_data.append(self.transform(data))
        self.num_task = labels.shape[1]

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = self.mol_data[idx]
        graph.label = torch.Tensor(self.labels[idx])
        graph.smi = self.smiles[idx]
        return graph
