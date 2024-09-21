def add_elements_to_pdb(input_pdb_file, output_pdb_file):
    with open(input_pdb_file, 'r') as f_in, open(output_pdb_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                element = atom_name[0].upper() if atom_name[0].isalpha() else atom_name[1].upper()
                line = line.rstrip() + ' ' * (78 - len(line)) + element + '\n'
            f_out.write(line)

input_pdb_file = 'dataset/cath40_k10_imem_add2ndstrc/pdb_format/'+'1a0gA02.pdb'
output_pdb_file = 'dataset/cath40_k10_imem_add2ndstrc/pdb_format/'+'1a0gA02_repair.pdb'
add_elements_to_pdb(input_pdb_file, output_pdb_file)
