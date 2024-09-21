import os
from dataset_src.cath_imem_2nd import Cath_imem,dataset_argument
from torch.optim import Adam
from torch_geometric.data import Batch,Data
from dataset_src.utils import NormalizeProtein
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import torch.nn.functional as F
import torch
from tqdm import tqdm
amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def get_struc2ndRes(pdb_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))
    p = PDBParser()
    structure = p.get_structure('', pdb_filename)
    model = structure[0]
    dssp = DSSP(model, pdb_filename,dssp='mkdssp')
    sec_structures = [dssp_res[2] for dssp_res in dssp]
    sec_structure_str = ''.join(sec_structures)
    sec_structure_str = sec_structure_str.replace('-','L')
    integer_encoded = [char_to_int[char] for char in sec_structure_str]
    data = F.one_hot(torch.tensor(integer_encoded), num_classes = 8)
    return data

def pdb2graph(dataset,filename):
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = dataset.get_receptor_inference(filename)
    struc_2nd_res = F.one_hot(torch.randint(0,8,size=(len(rec_coords),1)).squeeze(), num_classes = 8)#get_struc2ndRes(filename)
    rec_graph = dataset.get_calpha_graph(
                rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
    if rec_graph:
        normalize_transform = NormalizeProtein(filename='dataset/cath40_k10_imem_add2ndstrc/mean_attr.pt')
        
        graph = normalize_transform(rec_graph)
        return graph
    else:
        return None


#### dataset  ####
dataset_arg = dataset_argument(n=51)
dataset_arg['root'] = dataset_arg['root']
CATH_test_inmem = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
                            divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
                            c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
                            set_length=dataset_arg['set_length'],
                            struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
                            random_sampling=True,diffusion=True)

error_pdb = []
protein = 'GFP'
save_dir = 'dataset/TS/test_set/TS50/process/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
filename_list = os.listdir(f'dataset/TS/test_set/TS50/pdb/')
for filename in tqdm(filename_list[10:]):
    if os.path.exists(save_dir+filename.replace('.pdb','.pt')):
        pass
    else:
        graph = pdb2graph(CATH_test_inmem,f'dataset/TS/test_set/TS50/pdb/'+filename)
        if graph:
            torch.save(graph,save_dir+filename.replace('.pdb','.pt'))
        else:
            error_pdb.append(filename)

print(len(error_pdb))
print(error_pdb)

