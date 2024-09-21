import re
import os
import requests

def parse_a2m(file_content):
    sequences = {}
    seq_id = None
    seq = ""

    for line in file_content.split('\n'):
        if line.startswith('>'):
            if seq_id:
                sequences[seq_id] = seq

            match = re.search(r"\w+\|(\w+)", line)
            if match:
                seq_id = match.group(1)  # 提取Uniprot ID
                seq = ""
            else:
                print(f"Warning: Unable to extract Uniprot ID from: {line}")
                seq_id = None

        elif seq_id:
            seq += line.strip()

    if seq_id:
        sequences[seq_id] = seq

    return sequences



def download_alphafold_structure(uniprot_id, save_dir):
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
    response = requests.get(url)

    if response.status_code == 200:
        file_path = os.path.join(save_dir, f'AF-{uniprot_id}-F1-model_v4.pdb')
        with open(file_path, 'wb') as f:
            f.write(response.content)
            print(f"{uniprot_id} Structure saved to {save_dir}")
        return True
    else:
        print(f"Failed to download structure for {uniprot_id}. Error: {response.status_code}")
        return False


if __name__ == '__main__':
    # Read the .a2m file content
    finish_protein = ['BG_STRSQ','GFP']
    protein_list = os.listdir('dataset/MSA_structure/')
    for protein_name in protein_list:
        if protein_name not in finish_protein:
            a2m_file_path = f"dataset/MSA_structure/{protein_name}/{protein_name}.a2m"  # Replace with your .a2m file path
            with open(a2m_file_path, "r") as f:
                file_content = f.read()

            # Parse sequences from the .a2m file content
            sequence_dict = parse_a2m(file_content)

            # Download AlphaFold predicted structures and save to the specified output directory
            output_dir = f"dataset/MSA_structure/{protein_name}/raw/"
            already_download = [i.split('-')[1] for i in os.listdir(output_dir)]
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            attempted_downloads = 0
            successful_downloads = 0
            failed_downloads = 0

            for key in sequence_dict.keys():
                if key not in already_download:
                    attempted_downloads += 1
                    if download_alphafold_structure(key, output_dir):
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                else:
                    print(f'{key} already exist')
            


            with open(f"dataset/MSA_structure/{protein_name}/download_summary.txt", "w") as file:
                file.write("Download summary:\n")
                file.write(f"Total attempted downloads: {attempted_downloads}\n")
                file.write(f"Successful downloads: {successful_downloads}\n")
                file.write(f"Failed downloads: {failed_downloads}\n")

            print("Download summary written to download_summary.txt.")
