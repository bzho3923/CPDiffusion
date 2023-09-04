import random
import torch
import os
import argparse
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Batch,Data
from dataset.large_dataset import Cath
from run_pt import Trianer,EGNN_NET,Sparse_DIGRESS

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def seq_recovery(data,pred_seq):
    '''
    data.x is nature sequence

    '''
    recovery_list = []
    for i in range(data.ptr.shape[0]-1):
        nature_seq = data.x[data.ptr[i]:data.ptr[i+1],:].argmax(dim=1)
        pred = pred_seq[data.ptr[i]:data.ptr[i+1],:].argmax(dim=1)
        recovery = (nature_seq==pred).sum()/nature_seq.shape[0]
        recovery_list.append(recovery.item())
        ind = (nature_seq==pred)
    return recovery_list,ind


def compute_rr(seq1,seq2):
    count = 0
    for index,res in enumerate(seq1):
        if res == seq2[index]:
            count +=1
    return count


def creat_args():
    parser = argparse.ArgumentParser(description='protein diffusion')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file')
    parser.add_argument('--target_protein', type=str, default=None, help='target protein')
    parser.add_argument('--target_protein_dir', type=str, default=None)
    parser.add_argument('--fix_pos_file', type=str, default=None, help='fix position')
    parser.add_argument('--gen_num', type=int, default=100, help='number of generated sequences')
    parser.add_argument('--output_dir', type=str, default="result/Ago/predict/")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = creat_args()
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("using device",device)
    
    # get config    
    checkpoint = torch.load(args.ckpt)
    config = checkpoint['config']
    step = checkpoint['step']

    target_protein_dir = args.target_protein_dir
    target_protein_id = os.listdir(target_protein_dir)
    random.Random(4).shuffle(target_protein_id)
    target_protein_train_id, target_protein_test_id = target_protein_id[:-50], target_protein_id[-50:]
    print(target_protein_test_id)
    target_protein_train_dataset = Cath(target_protein_train_id, args.target_protein_dir)
    target_protein_test_dataset = Cath(target_protein_test_id, args.target_protein_dir)
    protein_name = args.target_protein.split("/")[-1].split(".")[0]
    
    print(f'train on {protein_name} dataset with {len(target_protein_train_dataset)}  training data and {len(target_protein_test_dataset)}  val data')
    input_feat_dim = target_protein_train_dataset[0].x.shape[1]+target_protein_train_dataset[0].extra_x.shape[1]
    edge_attr_dim = target_protein_train_dataset[0].edge_attr.shape[1]

    model = EGNN_NET(input_feat_dim=input_feat_dim,hidden_channels=config['hidden_dim'],edge_attr_dim=edge_attr_dim,dropout=config['drop_out'],n_layers=config['depth'],update_edge = True,embedding=config['embedding'],embedding_dim=config['embedding_dim'],norm_feat=config['norm_feat'],output_dim=20,embedding_ss=config['embed_ss'])
    diffusion = Sparse_DIGRESS(model=model,config=config,timesteps=config['timesteps'],objective=config['objective'],label_smooth_tem=config['smooth_temperature'])
    trainer  = Trianer(config,
                        diffusion,
                        target_protein_train_dataset, 
                        target_protein_test_dataset,
                        target_protein_test_dataset,
                        train_batch_size = 128,
                        gradient_accumulate_every=2,
                        save_and_sample_every=10,
                        train_num_steps=48000,
                        train_lr=config['lr'],
                        results_folder="result/weight/")

    relative_ckpt_path = "/".join(args.ckpt.split('/')[2:])
    trainer.load(milestone=9,filename=relative_ckpt_path)
    
    graph = torch.load(args.target_protein)
    data = Batch.from_data_list([graph])
    data_input = Data.clone(data)
    data_input.extra_x = torch.cat([data.x[:,20:],data.mu_r_norm],dim = -1)
    data_input.x = data.x[:,:20].to(torch.float32)
    print(data_input.x.shape)
    data_input.to('cuda:0')
    sub_recover_record = []
    sub_perplex_record=[]
    all_zt = []#torch.tensor([],device = device)
    count_seq = 0

    original_seq = ''.join([amino_acids_type[i.item()] for i in data_input.x.argmax(dim=1).cpu().numpy()])
    print(f'{args.target_protein} seq is:')
    print(original_seq)
    cond_index = [0]
    if args.fix_pos_file is not None:
        lines = open(args.fix_pos_file, "r").readlines()
        for line in lines:
            pos = int(line[1:]) - 1
            aa = line[0]
            assert original_seq[pos] == aa, f"original seq {original_seq[pos]} != {aa}"
            cond_index.append(pos)
    # cas12a 8
    # cond_index = torch.tensor([537, 594, 758, 767, 784, 831, 924, 1179])
    # cas9 21
    print(cond_index)
    cond_index = torch.tensor(cond_index)
    
    seq_record = {'id': [], 'seq': [], 'recovery': []}
    for i in range(args.gen_num):
        with torch.no_grad():
            zt,sample_graph = trainer.ema.ema_model.ddim_sample(data_input,cond=cond_index, temperature=1.0,stop=0,step=5) #zt is the output of Neural Netowrk and sample graph is a sample of it
            # prob = F.softmax(zt/10,dim=1)
            prob = zt.half()
            all_zt.append(prob) #= torch.cat([all_zt,zt])
            recovery_list,ind = seq_recovery(data_input,sample_graph)
            recovery = np.mean(recovery_list)
            
            #[B,L,37,hidden_dim] atom14 to atom37
            seq = ''.join([amino_acids_type[i.item()] for i in zt.argmax(dim=1).cpu().numpy()])
            
            seq_record['id'].append(i)
            seq_record['seq'].append(seq)
            seq_record['recovery'].append(recovery)
            print(i,seq,recovery)
            # if i >= 1: 
            #     all_zt_tensor = torch.stack(all_zt)
            #     recovery = (all_zt_tensor.mean(dim = 0).argmax(dim=1) == data_input.x.argmax(dim = 1)).sum()/(all_zt_tensor.mean(dim = 0).argmax(dim=1) == data_input.x.argmax(dim = 1)).shape[0]
            #     ll_fullseq = F.cross_entropy(all_zt_tensor.mean(dim = 0),data_input.x, reduction='mean').item()
            #     perplexity = np.exp(ll_fullseq)
            #     print(i,recovery.item(),perplexity)
            #     sub_recover_record.append(recovery.item())
            #     sub_perplex_record.append(perplexity)
    
    # torch.save(all_zt_tensor.mean(dim = 0),'dataset/Ago/predict/likelihood_step=100_1.pt')
    
    os.makedirs(args.output_dir,exist_ok=True)
    out_file = os.path.join(args.output_dir, f'{protein_name}-{step}.csv')
    pd.DataFrame(seq_record).to_csv(out_file, index=False)
    print('finish')