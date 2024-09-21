from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
p = PDBParser()
structure = p.get_structure("", 'dataset/Ago/sample_1.pdb')
model = structure[0]
dssp = DSSP(model, "dataset/Ago/sample_1.pdb",dssp='mkdssp')
sec_structures = [dssp_res[2] for dssp_res in dssp]
sec_structure_str = ''.join(sec_structures)
print(sec_structure_str)