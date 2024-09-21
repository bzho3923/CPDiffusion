import os
import Bio
import Bio.AlignIO
import numpy as np
import math
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#对每一个序列进行PSSM打分
def pssm(proseq,outdir):
    # 一个命令函数，依据pdb文件。使用blast生成pssm文件
    def command_pssm(content, output_file, pssm_file):
        os.system('psiblast \
                    -query %s \
                    -db /global/software/blast/database/nr \
                    -num_iterations 3 \
                    -out %s \
                    -out_ascii_pssm %s &' % (content, output_file, pssm_file))
    inputfile = open(proseq,'r')
    content = ''
    input_file = ''
    output_file = ''
    pssm_file = ''
    chain_name = []
    for eachline in inputfile:
        if '>' in eachline:
            if len(content):
                temp_file = open(outdir + '/fasta/' + chain_name,'w')
                temp_file.write(content)
                input_file = outdir + '/fasta/' + chain_name
                output_file = outdir + '/' + chain_name + '.out'
                pssm_file = outdir + '/' + chain_name + '.pssm'
                command_pssm(input_file, output_file,pssm_file)
                temp_file.close
            content = ''
            chain_name = eachline[1:5] + eachline[6:7]
        content +=  ''.join(eachline)
        #print content
        #print chain_name
    if len(content):
        temp_file = open(outdir + '/fasta/' + chain_name,'w')
        temp_file.write(content)
        input_file = outdir + '/fasta/' + chain_name
        output_file = outdir + '/out/' + chain_name + '.out'
        pssm_file = outdir + '/pssm/' + chain_name + '.pssm'
        command_pssm(input_file, output_file,pssm_file)
        temp_file.close
    inputfile.close()

    # 格式化pssm每行数据
    def formateachline(eachline):
        col = eachline[0:5].strip()
        col += '\t' + eachline[5:8].strip()
        begin = 9
        end = begin + 3
        for i in range(20):
            begin = begin
            end = begin + 3
            col += '\t' + eachline[begin:end].strip()
            begin = end
        col += '\n'
        return col

def BLOSUM_like_score_matrix(MSA_path, threshod_gap_ratio=0.5, show_gap_ratio_distribution=False):
    """
    gap_ratio：throw away columns whose gap ratios are higher than this variety
    """
    residue_index_map = {b'A':0, b'R':1, b'N':2, b'D':3, b'C':4,
                         b'Q':5, b'E':6, b'G':7, b'H':8, b'I':9,
                         b'L':10, b'K':11, b'M':12, b'F':13, b'P':14,
                         b'S':15, b'T':16, b'W':17, b'Y':18, b'V':19,
                         b'a': 0, b'r': 1, b'n': 2, b'd': 3, b'c': 4,
                         b'q': 5, b'e': 6, b'g': 7, b'h': 8, b'i': 9,
                         b'l': 10, b'k': 11, b'm': 12, b'f': 13, b'p': 14,
                         b's': 15, b't': 16, b'w': 17, b'y': 18, b'v': 19
                         }
    MSA = Bio.AlignIO.read(MSA_path, "fasta")
    align_array = np.array([list(rec) for rec in MSA], np.character).T
    global_statistics = [0] * 20
    global_pair_statistics = np.zeros([20, 20])
    gap_ratio_list = []
    valid_column_num = 0
    global_pair_total = 0
    global_residue_total = 0
    for column in align_array:
        gap_num = 0
        column_statistics = [0] * 20
        for residue in column:
            if residue == b'X' or residue == b'x':
                continue
            elif residue == b'.' or residue == b'-':
                gap_num += 1
            else:
                column_statistics[residue_index_map[residue]] += 1

        gap_ratio = gap_num/align_array.shape[1]
        gap_ratio_list.append(gap_ratio)
        if gap_ratio < threshod_gap_ratio:
            valid_column_num += 1
            global_pair_total += (align_array.shape[1]-gap_num)*(align_array.shape[1]-gap_num-1)/2
            global_residue_total += (align_array.shape[1]-gap_num)
            for i in range(20):
                global_statistics[i] += column_statistics[i]
            for i in range(20):
                for j in range(20):
                    if i == j:
                        global_pair_statistics[i][j] += column_statistics[i]*(column_statistics[i]-1)/2
                    else:
                        global_pair_statistics[i][j] += column_statistics[i]*column_statistics[j]
    if show_gap_ratio_distribution:
        s = pd.Series(np.array(gap_ratio_list))
        s.hist(bins=20, histtype='bar', align='mid', orientation='vertical', alpha=0.5)
        # plt.title("MSA序列中每一位点的 gap ratio 分布")
        plt.show()
    print('MSA shape', align_array.shape)
    print('valid_column_num', valid_column_num)
    global_pair_statistics /= global_pair_total
    global_statistics = [i/global_residue_total for i in global_statistics]
    # print('residue sum', torch.tensor(global_statistics).sum().item())
    # print('pair sum', global_pair_statistics.sum(axis=0).sum(axis=0).item())
    S = np.zeros([20, 20])
    for i in range(20):
        for j in range(20):
            if i == j:
                S[i][j] += round(2*math.log(global_pair_statistics[i][j]/(global_statistics[i]*global_statistics[j]), 2))
            else:
                S[i][j] += round(2*math.log(global_pair_statistics[i][j]/(2*global_statistics[i]*global_statistics[j]), 2))
    S_torch = torch.tensor(S)
    return S_torch

def MSA_retrieval(MSA_path):
    """
    gap_ratio：throw away columns whose gap ratios are higher than this variety
    """
    residue_index_map = {b'A':0, b'R':1, b'N':2, b'D':3, b'C':4,
                         b'Q':5, b'E':6, b'G':7, b'H':8, b'I':9,
                         b'L':10, b'K':11, b'M':12, b'F':13, b'P':14,
                         b'S':15, b'T':16, b'W':17, b'Y':18, b'V':19,
                         b'a': 0, b'r': 1, b'n': 2, b'd': 3, b'c': 4,
                         b'q': 5, b'e': 6, b'g': 7, b'h': 8, b'i': 9,
                         b'l': 10, b'k': 11, b'm': 12, b'f': 13, b'p': 14,
                         b's': 15, b't': 16, b'w': 17, b'y': 18, b'v': 19
                         }
    other_residue_type = [b'x', b'X', b'u', b'U', b'b', b'B', b'j', b'J', b'o', b'O', b'z', b'Z']
    MSA = Bio.AlignIO.read(MSA_path, "fasta")
    align_array = np.array([list(rec) for rec in MSA], np.character).T
    global_statistics = [0] * 20
    global_pair_statistics = np.zeros([20, 20])
    gap_ratio_list = []
    valid_column_num = 0
    global_pair_total = 0
    global_residue_total = 0
    global_statistics_torch = None
    for i, column in enumerate(align_array):
        if column[0] == b'.' or column[0] == b'-':
            continue
        gap_num = 0
        column_statistics = [0] * 20
        for residue in column:
            if residue in other_residue_type:
                continue
            elif residue == b'.' or residue == b'-':
                gap_num += 1
            else:
                column_statistics[residue_index_map[residue]] += 1
        column_statistics_torch = torch.tensor(column_statistics, dtype=torch.float).resize(1, 20)
        column_statistics_torch /= (align_array.shape[1]-gap_num)
        if global_statistics_torch is None:
            global_statistics_torch = column_statistics_torch
        else:
            global_statistics_torch = torch.cat([global_statistics_torch, column_statistics_torch], dim=0)
    print('retrieval MSA tensor size: ', global_statistics_torch.shape, 'tensor: ', global_statistics_torch)
    global_statistics_torch = global_statistics_torch.cuda(device=0)
    return global_statistics_torch


if __name__ == '__main__':
    MSA_path = "MSA_ProteinGym/MSA_files/A0A1I9GEU1_NEIME_full_11-26-2021_b08.a2m"
    S = BLOSUM_like_score_matrix(MSA_path, 0.5, True)
    BLOSUM = torch.load('dataset_src/blosum_substitute.pt')
    sns.heatmap(BLOSUM['original_score'], square=True)
    plt.show()
    sns.heatmap(S, square=True)
    plt.show()