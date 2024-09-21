import re
import os
import requests




# def parse_a2m(a2m_content,protein_name):
#     a2m_lines = a2m_content.strip().split('\n')
#     sequence_dict = {}

#     for line in a2m_lines :
#         if line.startswith('>') and '>'+protein_name+'/' not in line:
#             uniprot_id = re.search(r'\|([A-Za-z0-9]+)\|', line).group(1)
#             start, end = map(int, re.search(r'/(\d+)-(\d+)', line).groups())
#             sequence = next(iter(re.finditer('[a-zA-Z]+', line))).group(0)
#             sequence_dict[uniprot_id] = (sequence, start, end)

#     return sequence_dict



def parse_a2m(a2m_content):
    a2m_lines = a2m_content.strip().split('\n')
    sequence_dict = {}
    seq = []

    for line in a2m_lines:
        if line.startswith('>'):
            if seq:
                sequence = ''.join(seq)
                sequence_dict[uniprot_id] = (sequence, start, end)
                seq = []
            try:
                uniprot_id = re.search(r'\|([A-Za-z0-9]+)\|', line).group(1)
            except AttributeError:
                uniprot_id = re.search(r'>([A-Za-z0-9]+)', line).group(1)
            start, end = map(int, re.search(r'/(\d+)-(\d+)', line).groups())
        else:
            seq.append(line.strip())

    if seq:
        sequence = ''.join(seq)
        sequence_dict[uniprot_id] = (sequence, start, end)

    return sequence_dict


def modify_pdb(pdb_file, sequence, start, end, output_folder):
    with open(pdb_file, 'r') as file:
        pdb_lines = file.readlines()

    new_pdb_lines = []
    for line in pdb_lines:
        if line.startswith('ATOM'):
            residue_num = int(line[22:26].strip())

            if start <= residue_num <= end:
                new_pdb_lines.append(line)
        else:
            new_pdb_lines.append(line)

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, os.path.basename(pdb_file))
    with open(output_file, 'w') as file:
        file.writelines(new_pdb_lines)

def a2m_to_sequence(a2m_seq):
    sequence = ''
    for aa in a2m_seq:
        if aa.isupper() or aa == '-':
            sequence += aa
    return sequence


protein_name = 'BG_STRSQ'

save_directory = f"dataset/MSA_structure/{protein_name}/raw/"
out_dir = f"dataset/MSA_structure/{protein_name}/modified_pdb_files/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
a2m_file_path = f"dataset/MSA_structure/{protein_name}/{protein_name}.a2m"  # Replace with your .a2m file path
with open(a2m_file_path, "r") as f:
    file_content = f.read()

# Parse sequences from the .a2m file content
sequence_dict = parse_a2m(file_content)

# for key, a2m_seq in sequence_dict.items():
#     pdb_file = os.path.join(save_directory, f'AF-{key}-F1-model_v4.pdb')
#     if os.path.exists(pdb_file):
#         sequence = a2m_to_sequence(a2m_seq)
#         modify_pdb(pdb_file, sequence, out_dir)
#     else:
#         print(f"PDB file not found for {key}: {pdb_file}")

for key, value in sequence_dict.items():
    a2m_seq, start, end = value
    pdb_file = os.path.join(save_directory, f'AF-{key}-F1-model_v4.pdb')
    if os.path.exists(pdb_file):
        modify_pdb(pdb_file, a2m_seq, start, end, out_dir)
    else:
        print(f"PDB file not found for {key}: {pdb_file}")



