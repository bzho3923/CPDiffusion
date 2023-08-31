import os
import sys
import ultraimport
Cath = ultraimport('../dataset_src/cath_imem_2nd.py','Cath_imem')
# from dataset_src.cath_imem_2nd import Cath_imem as Cath

import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import dgd.utils as utils
from dgd.datasets.abstract_dataset import  AbstractDatasetInfos,AbstractDataModule
from dgd.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from dgd.analysis.rdkit_functions import compute_molecular_metrics


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]



class CATHDataModule(AbstractDataModule):
    def __init__(self, cfg,dataset_arg):
        # self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.dataset_arg = dataset_arg

    def prepare_data(self) -> None:
        datasets = {'train': Cath(self.dataset_arg['root'], self.dataset_arg['name'], split='test',
                              divide_num=self.dataset_arg['divide_num'], divide_idx=self.dataset_arg['divide_idx'],
                              c_alpha_max_neighbors=self.dataset_arg['c_alpha_max_neighbors'],
                              set_length=self.dataset_arg['set_length'],
                              struc_2nds_res_path = self.dataset_arg['struc_2nds_res_path'],
                              diffusion = self.dataset_arg['diffusion'],
                              pre_equivariant = self.dataset_arg['pre_equivariant']),
                    'val': Cath(self.dataset_arg['root'], self.dataset_arg['name'], split='val',
                              divide_num=self.dataset_arg['divide_num'], divide_idx=self.dataset_arg['divide_idx'],
                              c_alpha_max_neighbors=self.dataset_arg['c_alpha_max_neighbors'],
                              set_length=self.dataset_arg['set_length'],
                              struc_2nds_res_path = self.dataset_arg['struc_2nds_res_path'],
                              diffusion = self.dataset_arg['diffusion'],
                              pre_equivariant = self.dataset_arg['pre_equivariant']),
                    'test': Cath(self.dataset_arg['root'], self.dataset_arg['name'], split='test',
                              divide_num=self.dataset_arg['divide_num'], divide_idx=self.dataset_arg['divide_idx'],
                              c_alpha_max_neighbors=self.dataset_arg['c_alpha_max_neighbors'],
                              set_length=self.dataset_arg['set_length'],
                              struc_2nds_res_path = self.dataset_arg['struc_2nds_res_path'],
                              diffusion = self.dataset_arg['diffusion'],
                              pre_equivariant = self.dataset_arg['pre_equivariant'])}
        super().prepare_data(datasets)


class Cathinfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.name = 'cath'
        self.max_n_nodes = 1201
        self.number2secstruct = {0:'E', 1:'L', 2:'I', 3:'T', 4:'H', 5:'B', 6:'G', 7:'S'}
        self.secstruct2number = {value:key for key,value in self.number2secstruct.items()}
        