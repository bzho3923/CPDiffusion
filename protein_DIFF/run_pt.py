import math
import os
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import random 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.optim import Adam,AdamW
import torch_geometric
from torch_geometric.data import Batch,Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel

from tqdm.auto import tqdm
from ema_pytorch import EMA

# from accelerate import Accelerator

from dgd.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete
from model.egnn_pytorch.egnn_pyg_v2 import EGNN_Sparse
from model.egnn_pytorch.utils import nodeEncoder,edgeEncoder
# from dataset_src.cath_imem_2nd import Cath_imem,dataset_argument
from dataset.large_dataset import Cath
from dataset.utils import NormalizeProtein,substitute_label
from dataset.cath_imem_2nd import Cath_imem,dataset_argument

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from MSA_Transition_Matrix import MSA_retrieval

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any() or (tensor<0).any()

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_struc2ndRes(struc_2nds_res_filename):
    struc_2nds_res_alphabet = ['E', 'L', 'I', 'T', 'H', 'B', 'G', 'S']
    char_to_int = dict((c, i) for i, c in enumerate(struc_2nds_res_alphabet))

    if os.path.isfile(struc_2nds_res_filename):
        #open text file in read mode
        text_file = open(struc_2nds_res_filename, "r")
        #read whole file to a string
        data = text_file.read()
        #close file
        text_file.close()
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in data]
        print(len(data))
        data = F.one_hot(torch.tensor(integer_encoded), num_classes = 8)
        return data
    else:
        print('Warning: ' + struc_2nds_res_filename + 'does not exist')
        return None


def pdb2graph(dataset,filename,struc_2nd_res_file):
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = dataset.get_receptor_inference(filename)
    # struc_2nd_res_file = 'dataset/evaluation/DATASET/AMIE_PSEAE/ss'
    struc_2nd_res = get_struc2ndRes(struc_2nd_res_file)
    rec_graph = dataset.get_calpha_graph(
                rec, c_alpha_coords, n_coords, c_coords, rec_coords, struc_2nd_res)
    normalize_transform = NormalizeProtein(filename='dataset/cath40_k10_imem_add2ndstrc/mean_attr.pt')
    
    graph = normalize_transform(rec_graph)
    return graph


def prepare_mutation_graph(protein,dataset):
    '''
    Input DSM protein filename

    Output a list of graph, oringinal amino acid list, mutated amino acid list, mutation location list
    '''

    print('generative graph from pdb')
    filename = 'dataset/evaluation/DATASET/'+protein+'/'+protein+'.pdb'
    struc_2nd_res_file = 'dataset/evaluation/DATASET/'+protein+'/'+'ss'
    graph = pdb2graph(dataset,filename,struc_2nd_res_file)

    
    mutation_record_file = 'dataset/evaluation/DATASET/'+protein+'/'+protein+'.1.tsv' #.1 means single site mutation
    mutation_record = pd.read_csv(mutation_record_file, sep = '\t')

    type1, type2, location,score = [], [], [],[]
    for i in mutation_record.index:
        if int(mutation_record.loc[i,'mutant'][1:-1]) != graph.distances.shape[0] and '_' not in mutation_record.loc[i,'mutant']:#not record last position
            type1.append((mutation_record.loc[i,'mutant'][0]))
            location.append(int(mutation_record.loc[i,'mutant'][1:-1]))
            type2.append(mutation_record.loc[i,'mutant'][-1])
            score.append(mutation_record.loc[i,'score'])
    short_location = list(set(location)) 
    short_location.sort()
    graph_list = []
    for loc in short_location:
        graph_ = Data.clone(graph)
        graph_.mutation_pos = loc-1
        graph_list.append(graph_)
    
    return graph_list,type1,type2,location,score


@torch.no_grad()
def compute_single_site_corr_score_all(diffusion,dataset,corr_train_record,pred_sasa,stop=450):
    DSM_list = os.listdir('dataset/evaluation/DATASET')
    DSM_list.remove('.DS_Store')
    DSM_list.sort()
    corr_list = []
    count_list = []
    all_score_record = []
    for index,protein in enumerate(DSM_list):
        
        graph_list,type1,type2,location,score = prepare_mutation_graph(protein,dataset)
        short_loc = list(set(location))
        short_loc.sort()
        pred_list,pred_scratch_list =[], []
    
        pred_score_list,pred_score_scratch_list =[], []

        for realization in range(10):
            pred_score = []
            pred = torch.tensor([],device='cuda:0')
            graph = [graph_list[0]]
            data = Batch.from_data_list(graph)
            data_input = Data.clone(data)
            if pred_sasa:
                data_input.extra_x = torch.cat([data.x[:,22:],data.mu_r_norm],dim = 1)
            else:
                data_input.extra_x = torch.cat([data.x[:,20].unsqueeze(dim=1),data.x[:,22:],data.mu_r_norm],dim = 1)
            data_input.x = data.x[:,:20].to(torch.float32)
            data_input.to('cuda:0')            
            t_int = torch.ones(size=(data.batch[-1]+1, 1), device=data_input.x.device).float()*(500-stop)
            noise_data = diffusion.apply_noise(data_input ,t_int)
            pred,_ = diffusion.model(noise_data,t_int)
            pred_list.append(pred)

            # zt,sample_graph = diffusion.sample(data_input,1.0,stop=stop)
            pred_scratch_list.append(pred)
        averge_pred = torch.stack(pred_list).mean(dim = 0)#TODO fixed batch_size < number of graph bug
        averge_pred_scratch = torch.stack(pred_scratch_list).mean(dim = 0)
        print((averge_pred_scratch.argmax(dim =1).cpu() == data.x[:,:20].argmax(dim =1)).sum()/data.x.shape[0])
        print((averge_pred.argmax(dim =1).cpu() == data.x[:,:20].argmax(dim =1)).sum()/data.x.shape[0])
            # if realization%10 == 0:
            #     cat = torch.stack(pred_list).mean(dim = 0)[mutation_index[0]]
            #     plt.bar(list(range(20)),cat.cpu().detach().numpy())
            #     plt.savefig(f'protein_DIFF/results/sample_result/Feb_14th_posterior distribution on {realization} average with temperature{temperature}.png')
            #     plt.close()

        # all_data = Batch.from_data_list(graph_list).to('cuda:0')
        # mutation_index = all_data.ptr[:-1]+all_data.mutation_pos
        # acc = (averge_pred[mutation_index].argmax(dim=1) == all_data.x[mutation_index].argmax(dim=1) ).sum()/all_data.x[mutation_index].argmax(dim=1).shape[0]
        
        for ind,wild_type in enumerate(type1):
            # pred_scratch_score = averge_pred_scratch.cpu()[location[ind]-1][amino_acids_type.index(type2[ind])] - averge_pred_scratch.cpu()[location[ind]-1][amino_acids_type.index(type1[ind])]
            # pred_score = averge_pred.cpu()[location[ind]-1][amino_acids_type.index(type2[ind])] - averge_pred.cpu()[location[ind]-1][amino_acids_type.index(type1[ind])]
            
            target = data.x[:,:20].argmax(dim =1).clone()
            target[location[ind]-1] = amino_acids_type.index(type2[ind])
            pred_scratch_score = F.cross_entropy(averge_pred_scratch.cpu(),target)
            pred_score = F.cross_entropy(averge_pred.cpu(),target)
            pred_score_list.append(-pred_score.item())
            pred_score_scratch_list.append(-pred_scratch_score.item())

        corr = spearmanr(pred_score_list,score)[0]
        corr_scratch = spearmanr(pred_score_scratch_list,score)[0]
        print(protein,'step=',stop,corr,corr_scratch,len(type1))
    
        all_score_record.append([protein,stop,corr,corr_scratch])
        corr_list.append(np.abs(corr))
        count_list.append(len(type1))
        corr_train_record[index].append(np.abs(corr))

            # plt.plot(list(range(realization+1)),corr_list)
            # plt.savefig(f'protein_DIFF/results/sample_result/Feb_14th_corr_vs_num_of_sample__{realization} with temperature{temperature}.png',dpi = 200)
            # plt.title(protein+'spearman_corr')
            # plt.close()
    
    # torch.save(torch.tensor(corr_list),'corr_vs_step.pt')
    weight_average = 0
    for ind,corr in enumerate(corr_list):
        weight_average += count_list[ind]/sum(count_list) * corr_list[ind]
    return weight_average,corr_train_record,DSM_list


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.view(emb.shape[0],-1)

class EGNN_NET(torch.nn.Module):
    def __init__(self, input_feat_dim, hidden_channels, edge_attr_dim,  dropout, n_layers, output_dim = 20,
                 embedding=False, embedding_dim=64,update_edge = True,norm_feat = False,embedding_ss = False):
        super(EGNN_NET, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout

        self.update_edge = update_edge
        self.mpnn_layes = nn.ModuleList()
        self.time_mlp_list = nn.ModuleList()
        self.ff_list = nn.ModuleList()
        self.ff_norm_list = nn.ModuleList()
        self.sinu_pos_emb = SinusoidalPosEmb(hidden_channels)
        self.embedding = embedding
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.embedding_ss = embedding_ss

        self.time_mlp = nn.Sequential(self.sinu_pos_emb, nn.Linear(hidden_channels, hidden_channels),nn.SiLU(),
                                       nn.Linear(hidden_channels, embedding_dim))  

        self.ss_mlp = nn.Sequential(nn.Linear(8, hidden_channels), nn.SiLU(),
                                    nn.Linear(hidden_channels, embedding_dim))     

        for i in range(n_layers):
            if i == 0:
                layer = EGNN_Sparse(embedding_dim, m_dim=hidden_channels,hidden_dim=hidden_channels,out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge = self.update_edge,norm_feats=norm_feat)
            else: 
                layer = EGNN_Sparse(hidden_channels, m_dim=hidden_channels,hidden_dim=hidden_channels,out_dim=hidden_channels, edge_attr_dim=embedding_dim, dropout=dropout,
                                    update_edge = self.update_edge,norm_feats=norm_feat)                
            
            time_mlp_layer = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, (hidden_channels) * 2))
            ff_norm = torch_geometric.nn.norm.LayerNorm(hidden_channels)
            ff_layer = nn.Sequential(nn.Linear(hidden_channels, hidden_channels*4), nn.Dropout(p=dropout),nn.GELU(),nn.Linear(hidden_channels*4, hidden_channels)) 

            self.mpnn_layes.append(layer)
            self.time_mlp_list.append(time_mlp_layer)
            self.ff_list.append(ff_layer)
            self.ff_norm_list.append(ff_norm)


        if output_dim == 20:
            self.node_embedding = nodeEncoder(embedding_dim,feature_num=4)
        else:    
            self.node_embedding = nodeEncoder(embedding_dim,feature_num=3)

        self.edge_embedding = edgeEncoder(embedding_dim)
        self.lin = Linear(hidden_channels, output_dim)


    def forward(self, data,time): 
        #data.x first 20 dim is noise label. 21 to 34 is knowledge from backbone, e.g. mu_r_norm, sasa, b factor and so on
        x, pos, extra_x, edge_index, edge_attr,ss, batch = data.x, data.pos, data.extra_x, data.edge_index, data.edge_attr, data.ss,data.batch

        t = self.time_mlp(time)

        ss_embed = self.ss_mlp(ss)


        x = torch.cat([x,extra_x],dim=1)
        if self.embedding:
            x = self.node_embedding(x)
            edge_attr = self.edge_embedding(edge_attr)
        x = torch.cat([pos, x], dim=1)

        for i, layer in enumerate(self.mpnn_layes):
            #GNN aggregate
            if self.update_edge:
                h,edge_attr = layer(x, edge_index, edge_attr, batch) #[N,hidden_dim]
            else:
                h = layer(x, edge_index, edge_attr, batch) #[N,hidden_dim]
            
            #time and conditional shift
            corr, feats = h[:,0:3],h[:,3:]
            time_emb = self.time_mlp_list[i](t) #[B,hidden_dim*2]
            scale_, shift_ = time_emb.chunk(2,dim=1)
            scale = scale_[data.batch]
            shift = shift_[data.batch]
            feats = feats*(scale+1) +shift
            
            #FF neural network
            feature_norm = self.ff_norm_list[i](feats,batch)
            feats = self.ff_list[i](feature_norm) + feature_norm

            #TODO add skip connect
            x = torch.cat([corr, feats], dim=-1) 


        corr, x = x[:,0:3],x[:,3:]
        if self.embedding_ss:
            x = x+ss_embed 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        if self.output_dim == 21:
            return x[:,:20],x[:,20]
        else:
            return x, None


class BlosumTransition:
    def __init__(self, blosum_path='dataset_src/blosum_substitute.pt',x_classes=20,timestep = 500):
        self.original_score,self.temperature_list = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['temperature']
        self.X_classes = x_classes
        self.timestep = timestep
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        # if timestep == 500:
        #     self.temperature_list = self.temperature_list
        # else:
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        output_tensor = F.interpolate(temperature_list, size=timestep+1, mode='linear', align_corners=True)
        self.temperature_list = output_tensor.squeeze()


    
    def get_Qt_bar(self, time, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(time * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        return q_x


class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes


    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x.to(device)

        return q_x



class Sparse_DIGRESS(nn.Module):
    def __init__(self,model,config,*,timesteps=1000,sampling_timesteps = None,loss_type='CE',objective = 'pred_x0',beta_schedule = 'sigmoid',label_smooth_tem=1.0,schedule_fn_kwargs = dict()):
        super().__init__()
        self.model = model
        # self.self_condition = self.model.self_condition
        self.objective = objective
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.noise_type = config['noise_type']
        self.config  = config
        if config['noise_type'] == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=20)
        elif config['noise_type'] == 'blosum':
            self.transition_model = BlosumTransition(timestep=self.timesteps+1)
        
        self.label_smooth_tem = label_smooth_tem
        assert objective in {'pred_noise', 'pred_x0', 'pred_v','smooth_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(noise_schedule='cosine',timesteps=self.timesteps,noise_type=self.noise_type)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'CE':
            return F.cross_entropy

    def apply_noise(self,data,t_int):
        s_int = t_int - 1 
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps    

        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        device = data.x.device
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze()
        prob_X = prob_X/prob_X.sum(dim = 1, keepdim=True)

        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 20)
        noise_data = data.clone()
        noise_data.x = noise_X.to(data.x.device)
        
        return noise_data
    
    def sample_discrete_feature_noise(self,limit_dist ,num_node):
        x_limit = limit_dist[None,:].expand(num_node,-1) #[num_node,20]
        U_X = x_limit.flatten(end_dim=-2).multinomial(1).squeeze()
        U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
        return U_X

    def compute_batched_over0_posterior_distribution(self,X_t,Q_t,Qsb,Qtb,data):
        """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
        """
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Q_t.transpose(-1,-2)
        X_t_ = X_t.unsqueeze(dim = -2)
        left_term = X_t_ @ Qt_T[data.batch] #[N,1,d_t-1]
        # left_term = left_term.unsqueeze(dim = 1) #[N,1,dt-1]

        right_term = Qsb[data.batch] #[N,d0,d_t-1]

        numerator = left_term * right_term #[N,d0,d_t-1]

        prod = Qtb[data.batch] @ X_t.unsqueeze(dim=2) # N,d0,1
        denominator = prod
        denominator[denominator == 0] = 1e-6        

        out = numerator/denominator

        return out

    def compute_posterior_distribution(self,M_t, Qt_M, Qsb_M, Qtb_M,data):
        """ 
        M: is the distribution of X_0
        Compute  q(x_t-1|x_t,x_0) = xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0 
        """
         
        #X_t is a sample of q(x_t|x_t+1)
        Qt_T = Qt_M.transpose(-1,-2)
        X_t = M_t.unsqueeze(dim = -2)
        left_term = X_t @ Qt_T[data.batch] #[N,1,d_t-1]
        
        M_0 = data.x.unsqueeze(dim = -2) #[N,1,d_t-1]
        right_term = M_0@Qsb_M[data.batch] #[N,1,dt-1]
        numerator = (left_term * right_term).squeeze() #[N,d_t-1]


        X_t_T = M_t.unsqueeze(dim = -1)
        prod = M_0@Qtb_M[data.batch]@X_t_T # [N,1,1]
        denominator = prod.squeeze()
        denominator[denominator == 0] = 1e-6        

        out = (numerator/denominator.unsqueeze(dim=-1)).squeeze()

        return out        #[N,d_t-1]
    
    def sample_p_zs_given_zt(self,t,s,zt,data,temperature,last_step,cond=False):
        """
        sample zs~p(zs|zt)
        """
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)#check for this
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)#check for this

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, data.x.device)
        # Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        Qt = (Qtb/Qsb)/(Qtb/Qsb).sum(dim=1).unsqueeze(dim=2)

        noise_data = data.clone()   
        noise_data.x = zt #x_t
        pred,_ = self.model(noise_data,t*self.timesteps)
        pred_X = F.softmax(pred,dim = -1) #\hat{p(X)}_0
        
        if isinstance(cond, torch.Tensor):
            pred_X[cond] = data.x[cond]

        if last_step:
            pred = pred**temperature
            pred_X = F.softmax(pred,dim = -1)
            # sample_s = pred_X.multinomial(1).squeeze()
            sample_s = pred_X.argmax(dim = 1)
            final_predicted_X = F.one_hot(sample_s,num_classes = 20).float()

            return pred,final_predicted_X
            
        
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=zt,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        # prob_X = prob_X/temperature
        sample_s = prob_X.multinomial(1).squeeze()
        # sample_s = prob_X.argmax(1).squeeze()
        X_s = F.one_hot(sample_s,num_classes = 20).float()
        
        return X_s,final_predicted_X if last_step else None
    
    def sample_p_zs_given_zt_MSA(self,t,s,zt,data,temperature,last_step,cond=False, MSA_retrieval_ratio=0):
        """
        sample zs~p(zs|zt)
        """

        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)#check for this
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)#check for this

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, data.x.device)
        # Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        Qt = (Qtb/Qsb)/(Qtb/Qsb).sum(dim=1).unsqueeze(dim=2)

        noise_data = data.clone()   
        noise_data.x = zt #x_t
        pred,_ = self.model(noise_data,t*self.timesteps)
        pred_X = F.softmax(pred,dim = -1) #\hat{p(X)}_0
        
        if isinstance(cond, torch.Tensor):
            pred_X[cond] = data.x[cond]

        if last_step:
            pred = pred**temperature
            pred_X = F.softmax(pred,dim = -1)
            # add MSA distribution here
            # 序列输入到tmp/i.fasta
            MSA_pred_X = None
            seq_list = ['']*(1+data.batch[-1])
            for i, aa_type in enumerate(data.x.argmax(dim=1)):
                seq_list[data.batch[i]] += amino_acids_type[aa_type]
            for i, seq in enumerate(seq_list):
                print('seq len: ', len(seq), 'seq: ', seq)
                record = SeqRecord(Seq(seq), id=str(i))
                records = [record]
                fasta_file = 'protein_DIFF/test_rr_tmp/'+str(i)+'.fasta'
                MSA_a2m_file = 'protein_DIFF/test_rr_tmp/'+str(i)+'.a2m'
                SeqIO.write(records, fasta_file, 'fasta')
                if str(i)+'.a2m' not in os.listdir('protein_DIFF/test_rr_tmp/'):
                    p = os.popen('hhblits -i '+fasta_file+' -d /data/alphafold2/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt -d /data/alphafold2/data/pdb70/pdb70 -d /data/alphafold2/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08  -o example.hhr  -oa3m tmp.a3m  -n 3  -id 90  -cov 50  -cpu 8')
                    print(p.read())
                    p.close()
                    p = os.popen(' reformat.pl tmp.a3m '+MSA_a2m_file)
                    print(p.read())
                    p.close()
                MSA_pred_X_single_seq = MSA_retrieval(MSA_a2m_file)
                if MSA_pred_X is None:
                    MSA_pred_X = MSA_pred_X_single_seq
                else:
                    MSA_pred_X = torch.cat([MSA_pred_X, MSA_pred_X_single_seq], dim=0)
            print('MSA_pred_X shape: ', MSA_pred_X.shape)
            print('MSA_pred_X: ', MSA_pred_X)
            # os.removedirs('protein_DIFF/test_rr_tmp/')
            # os.makedirs('protein_DIFF/test_rr_tmp/')

            if type(MSA_retrieval_ratio) is list:
                final_predicted_X_list = []
                for ratio in MSA_retrieval_ratio:
                    mixed_pred_X = (1-ratio)*pred_X + ratio*MSA_pred_X
                    sample_s = mixed_pred_X.argmax(dim=1)
                    final_predicted_X = F.one_hot(sample_s, num_classes=20).float()
                    # final_predicted_X_list.append(final_predicted_X.clone())
                    final_predicted_X_list.append(final_predicted_X)
                return pred, final_predicted_X_list

            else:
                # sample_s = pred_X.multinomial(1).squeeze()
                mixed_pred_X = (1 - MSA_retrieval_ratio) * pred_X + MSA_retrieval_ratio * MSA_pred_X
                sample_s = mixed_pred_X.argmax(dim=1)
                final_predicted_X = F.one_hot(sample_s,num_classes = 20).float()

                return pred,final_predicted_X
            
        
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=zt,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        # prob_X = prob_X/temperature
        sample_s = prob_X.multinomial(1).squeeze()
        # sample_s = prob_X.argmax(1).squeeze()
        X_s = F.one_hot(sample_s,num_classes = 20).float()
        
        return X_s,final_predicted_X if last_step else None
    
    def sample(self,data,cond = False,temperature=1.0,stop = 0):
        limit_dist = torch.ones(20)/20
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in reversed(range(stop, self.timesteps)): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + 1
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,temperature,last_step=s_int==stop)
        return zt,final_predicted_X

    def ddim_sample(self,data,cond = False,temperature=1.0,stop = 0,step=10, ratio=0):
        limit_dist = torch.ones(20)/20
        zt = self.sample_discrete_feature_noise(limit_dist = limit_dist,num_node = data.x.shape[0]) #[N,20] one hot 
        zt = zt.to(data.x.device)
        for s_int in tqdm(list(reversed(range(stop, self.timesteps,step)))): #500
            #z_t-1 ~p(z_t-1|z_t),
            s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
            t_array = s_array + step
            s_norm = s_array / self.timesteps
            t_norm = t_array /self.timesteps
            if ratio != 0:
                zt , final_predicted_X  = self.sample_p_zs_given_zt_MSA(t_norm, s_norm,zt, data,temperature,last_step=s_int==stop,cond=cond, MSA_retrieval_ratio=ratio)
            else:
                zt , final_predicted_X  = self.sample_p_zs_given_zt(t_norm, s_norm,zt, data,temperature,last_step=s_int==stop,cond=cond)
        
        
        return zt,final_predicted_X
    

    def kl_prior(self,data):
        '''
        compute kl distance between prior distribution p(x_0) with predefined schedule distribution q(xT|x0)
        '''

        t_float = torch.ones(size=(data.batch[-1]+1, 1), device=data.x.device).float()
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze() #q(xT|x0) [N,20]

        limit_dist = torch.ones(20)/20
        num_node = data.x.shape[0]
        x_limit = limit_dist[None,:].expand(num_node,-1).type_as(prob_X)
        kl_distance_X = F.kl_div(input=prob_X.log(), target=x_limit, reduction='batchmean')
        
        return kl_distance_X

    def reconstruction_logp(self,data):
        # t_int = torch.zeros(size=(data.batch[-1]+1, 1),dtype=data.x.dtype).float()
        # beta_t = self.noise_schedule(t_normalized=t_float) 
        # Q_0 = self.transition_model.get_Qt(beta_t=beta_t,device=data.x.device)


        # alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        # Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        # print(Qtb[0,0],Q_0[0,0])
        if type(data) is list:
            data = Batch.from_data_list(data)
        device = next(self.model.parameters())
        data = data.to(device)
        t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        noise_data = self.apply_noise(data ,t_int)
        pred_X,_ = self.model(noise_data,t_int) #have parameter
        loss = self.loss_fn(pred_X,data.x,reduction='mean')
        return loss

    def diffusion_loss(self,data,t_int):
        '''
        Compute the divergence between  q(x_t-1|x_t,x_0) and p_{\theta}(x_t-1|x_t)
        
        '''
        # q(x_t-1|x_t,x_0)
        s_int = t_int - 1 
        t_float = t_int / self.timesteps
        s_float = s_int / self.timesteps    
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=data.x.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=data.x.device)
        Qt = self.transition_model.get_Qt(beta_t, data.x.device)
        prob_X = (Qtb[data.batch]@data.x[:,:20].unsqueeze(2)).squeeze()       
        X_t = prob_X.multinomial(1).squeeze()
        noise_X = F.one_hot(X_t,num_classes = 20).type_as(data.x)
        prob_true = self.compute_posterior_distribution(noise_X,Qt,Qsb,Qtb,data)  #[N,d_t-1]


        #p_{\theta}(x_t-1|x_t) = \sum_{x0} q(x_t-1|x_t,x_0)p(x0|xt)
        noise_data = data.clone()
        noise_data.x = noise_X #x_t
        t = t_int*torch.ones(size=(data.batch[-1]+1, 1), device=data.x.device).float()
        pred,_ = self.model(noise_data,t)
        pred_X = F.softmax(pred,dim = -1) #\hat{p(X)}_0
        p_s_and_t_given_0_X = self.compute_batched_over0_posterior_distribution(X_t=noise_X,Q_t=Qt,Qsb=Qsb,Qtb=Qtb,data=data)#[N,d0,d_t-1] 20,20
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X #[N,d0,d_t-1]
        unnormalized_prob_X = weighted_X.sum(dim=1)             #[N,d_t-1]
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_pred = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  #[N,d_t-1]
        # loss = F.kl_div(input=prob_pred.log(), target=prob_true, reduction='mean')
        loss = self.loss_fn(prob_pred,prob_true,reduction='mean')
        return loss

    def compute_val_loss(self,data,evaluate_all=False):
        # 1. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(data)
        # 2. 
        if evaluate_all:
            diffusion_loss_list= []
            for s_int in reversed(range(0, self.timesteps)): #500

                s_array = s_int * torch.ones((data.batch[-1]+1, 1)).type_as(data.x)
                Lt= self.diffusion_loss(data,s_array)
                diffusion_loss_list.append(Lt.item())
            diffusion_loss = sum(diffusion_loss_list)
        else:
            t_int = torch.randint(0, self.timesteps + 1, size=(data.batch[-1]+1, 1), device=data.x.device).float()
            diffusion_loss = self.diffusion_loss(data,t_int)
        # 3. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(data)

        total_val_loss = kl_prior+diffusion_loss+prob0

        if evaluate_all:
            return total_val_loss,diffusion_loss_list
        else:
            return total_val_loss,kl_prior,diffusion_loss,prob0
    

    def forward(self,data,logit=False):

        t_int = torch.randint(0, self.timesteps, size=(data.batch[-1]+1, 1), device=data.x.device).float()
        noise_data = self.apply_noise(data ,t_int)
        pred_X,pred_sasa = self.model(noise_data,t_int) #have parameter

        if self.objective == 'pred_x0':
            target = data.x
        elif self.objective == 'smooth_x0':
            target = substitute_label(data.x.argmax(dim = 1),temperature=self.label_smooth_tem)
        else:
            raise ValueError(f'unknown objective {self.objective}')
        ce_loss = self.loss_fn(pred_X,target,reduction='mean')
        
        if exists(pred_sasa):
            mse_loss = F.mse_loss(pred_sasa,data.sasa)
            loss = ce_loss + self.config['sasa_loss_coeff']*mse_loss
        else:
            loss = ce_loss

        if logit:
            return loss, pred_X
        else:
            if exists(pred_sasa):
                return loss, ce_loss,self.config['sasa_loss_coeff']*mse_loss    
            else:

                return loss, ce_loss,None



def seq_recovery(data,pred_seq):
    '''
    data.x is nature sequence

    '''
    recovery_list = []
    for i in range(data.ptr.shape[0]-1):
        nature_seq = data.x[data.ptr[i]:data.ptr[i+1],:].argmax(dim=1)
        # print('nature_seq', data.x[data.ptr[i]:data.ptr[i+1],:])
        pred = pred_seq[data.ptr[i]:data.ptr[i+1],:].argmax(dim=1)
        # print('pred_seq', pred_seq[data.ptr[i]:data.ptr[i+1],:])
        recovery = ((nature_seq==pred).sum()/nature_seq.shape[0])
        recovery_list.append(recovery.item())

    return recovery_list


class Trianer(object):
    def __init__(
        self,
        config,
        diffusion_model,
        train_dataset,
        val_dataset,
        test_dataset,
        *,
        train_batch_size = 512,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        weight_decay = 1e-2,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,#0.999
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 10000,
        num_samples = 25,
        results_folder = './result',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):    
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model = DataParallel(diffusion_model).to(device)
            # self.model = diffusion_model.to(device)
        else:
            self.model = diffusion_model.to(device)
            

        # self.model = diffusion_model
        self.config = config
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        self.ds = train_dataset
        dl = DataListLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 12)

        self.dl = cycle(dl)

        self.val_loader = DataLoader(val_dataset,batch_size=train_batch_size,shuffle=False, pin_memory = True, num_workers = 12)
        self.test_loader = DataLoader(test_dataset,batch_size=train_batch_size,shuffle=False, pin_memory = True, num_workers = 12)
        # optimizer

        self.opt = AdamW(diffusion_model.parameters(), lr = train_lr, betas = adam_betas,weight_decay= weight_decay)

        # for logging results in a folder periodically

        # if self.accelerator.is_main_process:
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        Path(results_folder+'/weight/').mkdir(exist_ok = True)
        Path(results_folder+'/figure/').mkdir(exist_ok = True)
        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.save_file_name = self.config['Date']+f"_lr={self.config['lr']}_wd={self.config['wd']}_dp={self.config['drop_out']}_hidden={self.config['hidden_dim']}_noisy_type={self.config['noise_type']}_embed_ss={self.config['embed_ss']}"
    
    def save(self, milestone):
        # if not self.accelerator.is_local_main_process:
        #     return
        if len(self.model.device_ids)>1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        data = {
            'config': self.config,
            'step': self.step,
            'model': state_dict,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            # 'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, os.path.join(str(self.results_folder),'weight', self.save_file_name+f'_{milestone}.pt'))
    
    def load(self, milestone,filename =False):
        # accelerator = self.accelerator
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"loading from {self.results_folder}")
        if filename:
            # data = torch.load(str(self.results_folder)+'/'+filename, map_location=device)
            data = torch.load(filename, map_location=device)
        else:
            data = torch.load(str(self.results_folder / self.config['Date']+f"model_lr={self.config['lr']}_dp={self.config['drop_out']}_timestep={self.config['timesteps']}_hidden={self.config['hidden_dim']}_{milestone}.pt"), map_location=device)

        # model = self.accelerator.unwrap_model(self.model)
        # clean_dict = {}
        # for key,value in data['model'].items():
        #     clean_dict[key.replace('module.','')] = value
            # model.state_dict()[key.replace('module.','')] = value
        self.model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # if exists(self.accelerator.scaler) and exists(data['scaler']):
        #     self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        lr_schedule = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        train_loss,ce_loss_list,mse_loss_list,recovery_list,perplexity,val_loss_list,corr_record= [],[],[],[],[],[],[]
        total_loss,total_ce_loss,total_mse_loss = 5,5,5
        corr_list = [ [] for i in range(28)]
        val_loss = torch.tensor([5.0])
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                for _ in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    data = next(self.dl)
                    loss,ce_loss,mse_loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    ce_loss = ce_loss / self.gradient_accumulate_every
                    total_loss += loss.mean().item()
                    total_ce_loss += ce_loss.mean().item()
                    if self.config['pred_sasa']: 
                        total_mse_loss += mse_loss.mean().item()

                    loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
                all_iter = len(self.ds)//self.batch_size
                num_iter = self.step % all_iter +1
                pbar.set_description(f'loss: {total_loss/num_iter:.4f}')

                if self.step%(len(self.ds)//self.batch_size) == 0 and self.step != 0:
                    
                    train_loss.append(total_loss/all_iter)
                    ce_loss_list.append(total_ce_loss/all_iter)
                    if self.config['pred_sasa']:
                        mse_loss_list.append(total_mse_loss/all_iter)
                    val_loss_list.append(val_loss.item())
                    total_loss = 0
                    total_ce_loss = 0
                    total_mse_loss = 0

                self.opt.step()
                self.opt.zero_grad()


                self.step += 1
                # if self.step > self.train_num_steps*3/4 and lr_schedule:
                #     for g in self.opt.param_groups:
                #         g['lr'] = self.config['lr']*0.1
                #     lr_schedule = False
                # elif self.step <5000:#warm up
                #     for g in self.opt.param_groups:
                #         g['lr'] = 1e-7 
                # else:
                #     for g in self.opt.param_groups:
                #         g['lr'] = self.config['lr']                         

                self.ema.to(device)
                self.ema.update()

                if self.step != 0 and self.step % (self.save_and_sample_every*(len(self.ds)//self.batch_size)) == 0:
                    self.ema.ema_model.eval()

                    with torch.no_grad():
                        sub_list = []
                        for data in self.val_loader:
                            data = data.to(device)
                            
                            # _,predx = self.model(data,logit=True)
                            # _,ema_pred = self.ema.ema_model.forward(data)
                            # val_loss,_,_,_ = self.ema.ema_model.compute_val_loss(data,False)
                            val_loss,ce_loss,mse_loss = self.ema.ema_model(data)

                            zt,sample_graph = self.ema.ema_model.ddim_sample(data,self.config['sample_temperature'],stop = 0,step=100) #zt is the output of Neural Netowrk and sample graph is a sample of it
                            recovery = np.mean(seq_recovery(data,sample_graph))
                            sub_list.append(recovery)
                            
                            ll_fullseq = F.cross_entropy(zt,data.x, reduction='mean').item()
                            # print(f'perplexity : {np.exp(ll_fullseq):.2f}')
                            
                            # _,all_t_val_loss = self.ema.ema_model.compute_val_loss(data,True)

                        # weigth_corr,corr_list,DSM_list = compute_single_site_corr_score_all(self.ema.ema_model,CATH_test_inmem,corr_list,self.config['pred_sasa'])
                        # corr_record.append(weigth_corr)

                        milestone = self.step // self.save_and_sample_every

                    recovery_list.append(np.mean(sub_list))
                    print(f'recovery rate is {np.mean(sub_list)}')
                    perplexity.append(0.1*np.exp(ll_fullseq)) 

                    fig = plt.figure(figsize=(8, 6))  
                    gs = GridSpec(3, 2, figure=fig)

                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.plot(train_loss, label='train_loss')
                    ax1.plot(val_loss_list, label='val_loss')
                    # ax1.plot(ce_loss_list,label = 'ce_loss')
                    if self.config['pred_sasa']:
                        ax1.plot(mse_loss_list,label = 'mse_loss')
                    ax1.set_ylim((0, 4))
                    ax1.legend(loc="upper right", fancybox=True)

                    ax2 = fig.add_subplot(gs[1, 0])
                    ax2.plot(recovery_list, label='recovery')
                    ax2.plot(perplexity, label='perplexity * 0.1')
                    ax2.legend(loc="upper right", fancybox=True)
                    ax2.set_title(f'best_recovery={max(recovery_list):.4f} at {recovery_list.index(max(recovery_list))}')
                    ax2.set_ylim((0, 0.8))


                    # ax4 = fig.add_subplot(gs[2, 0])
                    # ax4.plot(corr_record)
                    # ax4.set_title(f'best_corr={max(corr_record):.4f} at {corr_record.index(max(corr_record))}')

                    # ax3 = fig.add_subplot(gs[:, 1])
                    # for corr, protein_name in zip(corr_list, DSM_list):
                    #     ax3.plot(corr, label=protein_name)

                    # ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, prop={'size': 8})  
                    

                    plt.subplots_adjust(wspace=0.5, hspace=0.5)
                    plt.savefig(os.path.join(str(self.results_folder), 'figure', self.save_file_name + f'.png'), dpi=200)
                    plt.close()


                    # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                    self.save(milestone)

                pbar.update(1)
        train_detail = {'train_loss':train_loss,'val_loss':val_loss_list,'recovery':recovery_list,'perplexity':perplexity}
        pd.DataFrame(train_detail).to_csv(os.path.join(str(self.results_folder), self.save_file_name + f'.csv'), index=False)
        print('training complete')
    
    def test(self, ratio=0):
        """
        !! please delete all files in /test_rr_tmp if test set is changed.
        ratio: if input a list, plot a rr-ratio figure; if input a number, print recovery rate
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        recovery_list, perplexity = [], []
        self.ema.ema_model.eval()
        with torch.no_grad():
            sub_list = []
            ratio_sub_list = [[] for i in range(11)]
            for data in self.test_loader:
                print('data batch:', data.batch)
                data = data.to(device)
                
                # _,predx = self.model(data,logit=True)
                # _,ema_pred = self.ema.ema_model.forward(data)
                # val_loss,_,_,_ = self.ema.ema_model.compute_val_loss(data,False)
                val_loss,ce_loss,mse_loss = self.ema.ema_model(data)

                zt,sample_graph = self.ema.ema_model.ddim_sample(data,self.config['sample_temperature'],stop = 0,step=100, ratio=ratio) #zt is the output of Neural Netowrk and sample graph is a sample of it
                if type(sample_graph) is list:
                    # print('sample_graph: ', sample_graph)
                    for i, sample_graph_at_given_ratio in enumerate(sample_graph):
                        print('final prediction', i, sample_graph_at_given_ratio)
                        recovery = np.mean(seq_recovery(data, sample_graph_at_given_ratio))
                        print('recovery', recovery)
                        ratio_sub_list[i].append(recovery)
                else:
                    recovery = np.mean(seq_recovery(data,sample_graph))
                    sub_list.append(recovery)
                
                # print(f'perplexity : {np.exp(ll_fullseq):.2f}')
                
                # _,all_t_val_loss = self.ema.ema_model.compute_val_loss(data,True)

            # weigth_corr,corr_list,DSM_list = compute_single_site_corr_score_all(self.ema.ema_model,CATH_test_inmem,corr_list,self.config['pred_sasa'])
            # corr_record.append(weigth_corr)

        # recovery_list.append(np.mean(sub_list))
        if not sub_list == []:
            print(f'recovery rate is {np.mean(sub_list)}')
        else:
            # print('ratio_sub_list: ', ratio_sub_list)
            ratio_recovery_rate = [np.mean(l) for l in ratio_sub_list]
            print('ratio recovery rate: ', ratio_recovery_rate)
            plt.figure()
            plt.plot(ratio, ratio_recovery_rate)
            plt.xlabel("MSA_retrieval ratio")
            plt.ylabel("recovery rate")
            plt.savefig("recovery_rate.png")


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('--Date', type = str,default='Debug',
                        help='Date of experiment')
                    
    parser.add_argument('--data_split', action='store_true', default=False) 
    parser.add_argument('--cath_dir', type=str, default="dataset/cath40_k10_imem_add2ndstrc/process/")
     
    parser.add_argument('--objective', type = str,default='pred_x0',
                        help='the target of training objective, objective must be either pred_x0 or smooth_x0')  
    
    parser.add_argument('--lr', type = float,default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--smooth_temperature', type = float,default=1.0,
                        help='the temperature used for smoothing label')    
    
    parser.add_argument('--wd', type = float,default=1e-2,
                        help='weight decay')
    
    parser.add_argument('--drop_out', type = float,default=0.0,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--timesteps',  type = int,default=700,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--hidden_dim',  type = int,default=16,
                        help='Whether to run with best params for cora. Overrides the choice of dataset')

    parser.add_argument('--device_id', type = int,default=0,
                        help='cuda device')
    
    parser.add_argument('--sasa_loss_coeff',  type = float,default=1.0,
                        help='the coeff of mse of sasa prediction')

    parser.add_argument('--sample_temperature', type = float,default=1.0,
                        help='the temperature of predictive probability when t = 0')
    
    parser.add_argument('--depth', type = int,default=1,
                        help='number of GNN layers')
    
    parser.add_argument('--noise_type', type = str,default='uniform',
                        help='what type of noise apply in diffusion process, uniform or blosum')
    
    parser.add_argument('--embedding_dim', type = int,default=16,
                        help='the dim of feature embedding')   
    
    parser.add_argument('--clip_grad_norm',  type = float,default=1.0,
                        help='clip_grad_norm')

    parser.add_argument('--sampling_method',  type = str,default='ddpm',
                        help='clip_grad_norm')    

    parser.add_argument('--pred_sasa', action='store_true',#default = False,
                        help='whether predict sasa for better latent representation for mutation') 

    parser.add_argument('--embedding', action='store_true',#default = True,
                        help='whether residual embedding the feature') 
    
    parser.add_argument('--norm_feat', action='store_true',#default = True,
                        help='whether normalization node feature in egnn')    
    
    parser.add_argument('--embed_ss', action='store_true',#default = True,
                        help='whether embedding secondary structure')
    
    parser.add_argument('--batch_size', type = int,default=32)
    
    parser.add_argument('--target_protein_dir', type=str, default=None)
    
    parser.add_argument('--output_dir', type = str,default='./protein_DIFF/results',
                        help='output dir')
    
    args = parser.parse_args()
    config = vars(args)

    os.makedirs(config['output_dir'], exist_ok=True)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_id'])

    # config['pred_x0']
    print('train on CATH dataset')
    # dataset_arg = dataset_argument(n=51)
    # dataset_arg['root'] = 'protein_DIFF/'+dataset_arg['root']
    # train_dataset_inmem = Cath_imem(dataset_arg['root'], dataset_arg['name'], split='test',
    #                             divide_num=dataset_arg['divide_num'], divide_idx=dataset_arg['divide_idx'],
    #                             c_alpha_max_neighbors=dataset_arg['c_alpha_max_neighbors'],
    #                             set_length=dataset_arg['set_length'],
    #                             struc_2nds_res_path = dataset_arg['struc_2nds_res_path'],
    #                             random_sampling=True,diffusion=True)

    basedir = args.cath_dir
    if args.data_split:
        data_split  = torch.load('dataset/cath43_train_test_split.pt')
        train_idx, val_idx, test_idx = data_split['train'],data_split['validation'],data_split['test']#single_chain_id,L100.test
    else:
        cath_idx = os.listdir(basedir)
        random.Random(4).shuffle(cath_idx)
        # split train, val, test
        train_idx, val_idx, test_idx = cath_idx[1000:], cath_idx[:500], cath_idx[500:100]
            
    train_dataset = Cath(train_idx, basedir)
    val_dataset = Cath(val_idx, basedir)
    test_dataset = Cath(test_idx, basedir)
    print(f'train on CATH dataset with {len(train_dataset)}  training data and {len(val_dataset)}  val data')

    if args.target_protein_dir is not None:
        # example: ago_dir = 'dataset/Ago/process/'
        target_protein_name = args.target_protein_dir.split('/')[1]
        target_protein_id = os.listdir(args.target_protein_dir)
        target_protein_id.sort()
        random.Random(4).shuffle(target_protein_id)
        target_protein_train_id, target_protein_test_id = target_protein_id[:-50], target_protein_id[-50:]
        target_protein_train_dataset = Cath(target_protein_train_id, args.target_protein_dir)
        target_protein_test_dataset = Cath(target_protein_test_id, args.target_protein_dir)
        print(f'train on {target_protein_name} dataset with {len(target_protein_train_dataset)}')
        print(f'test on {target_protein_name} dataset with {len(target_protein_test_dataset)}')
        # ago_dataset = Cath(ago_id,ago_dir,pred_sasa=False)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, target_protein_train_dataset])
        val_dataset = target_protein_test_dataset
        test_dataset = target_protein_test_dataset
        
    input_feat_dim = train_dataset[0].x.shape[1]+train_dataset[0].extra_x.shape[1]
    edge_attr_dim = train_dataset[0].edge_attr.shape[1]

    if config['pred_sasa']:
        model = EGNN_NET(input_feat_dim=input_feat_dim,hidden_channels=config['hidden_dim'],edge_attr_dim=edge_attr_dim,dropout=config['drop_out'],n_layers=config['depth'],update_edge = True,embedding=config['embedding'],embedding_dim=config['embedding_dim'],norm_feat=config['norm_feat'],output_dim=21,embedding_ss=config['embed_ss'])
    else:
        model = EGNN_NET(input_feat_dim=input_feat_dim,hidden_channels=config['hidden_dim'],edge_attr_dim=edge_attr_dim,dropout=config['drop_out'],n_layers=config['depth'],update_edge = True,embedding=config['embedding'],embedding_dim=config['embedding_dim'],norm_feat=config['norm_feat'],embedding_ss=config['embed_ss'])#GVP,Protein_MPNN,
    
    # model = DataParallel(model)
    
    diffusion = Sparse_DIGRESS(model=model,config=config,timesteps=config['timesteps'],objective=config['objective'],label_smooth_tem=config['smooth_temperature'])

    trainer  = Trianer(config,
                        diffusion,
                        train_dataset, 
                        val_dataset,
                        test_dataset,
                        train_batch_size = args.batch_size,
                        gradient_accumulate_every=1,
                        save_and_sample_every=1,
                        train_num_steps=90000,
                        train_lr=config['lr'],
                        weight_decay = config['wd'],
                        results_folder=config['output_dir'])
    trainer.train()
    

# python3 protein_DIFF/diffusion_sequence_protein.py --lr 5e-4 --timesteps 500 --hidden_dim 128 --wd 1e-5 --clip_grad_norm 1e3 --device_id 0 --depth 6 --drop_out 0.1 --norm_feat --embedding_dim 128 --Date 'May14_Debug' --objective 'pred_x0' --smooth_temperature 0.9



