import os
def parse_pdb_confidence(pdb_content):
    atom_lines = [line for line in pdb_content.splitlines() if line.startswith('ATOM')]
    confidence_values = []

    for line in atom_lines:
        b_factor = float(line[60:66].strip())
        confidence_values.append(b_factor)

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    return average_confidence

# pdb_directory= 'dataset/1ud9A00_prediction/'
# pdb_file = 'low_rr.pdb'
# with open(os.path.join(pdb_directory, pdb_file), 'r') as f:
#     pdb_content = f.read()
# average_confidence = parse_pdb_confidence(pdb_content)
# print(average_confidence)

#84.13
#83.02
#83.5383


# protein_name = 'BG_STRSQ'

# save_directory = f"dataset/MSA_structure/{protein_name}/raw/"
# pdb_directory = f"dataset/MSA_structure/{protein_name}/modified_pdb_files/"

average_confidence_dict = {}



for i in range(1,694):
    pdb_dir = 'dataset/1_Ago_structure_database/AGO_summary/AGO_{:03}/unrelaxed_model_1_ptm.pdb'.format(i)
    with open(os.path.join(pdb_dir), 'r') as f:
            pdb_content = f.read()
    average_confidence = parse_pdb_confidence(pdb_content)
    average_confidence_dict[i] = average_confidence
    
print(average_confidence_dict)
# for pdb_file in os.listdir(pdb_directory):
#     if pdb_file.endswith('.pdb'):
#         with open(os.path.join(pdb_directory, pdb_file), 'r') as f:
#             pdb_content = f.read()
#         average_confidence = parse_pdb_confidence(pdb_content)
#         average_confidence_dict[pdb_file] = average_confidence

# print(average_confidence_dict)