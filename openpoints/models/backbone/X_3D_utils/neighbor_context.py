import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean,scatter_max,scatter_sum,scatter_softmax


class Neighbor_Context(nn.Module):
    def __init__(self,inchannel):
        super().__init__()
        conv_inchannel=inchannel*2
        self.convs=nn.Sequential(
                nn.Conv1d(conv_inchannel,inchannel,kernel_size=1),
                nn.BatchNorm1d(inchannel),
                nn.ReLU(),
                nn.Conv1d(inchannel,inchannel,kernel_size=1),
                nn.BatchNorm1d(inchannel),
                nn.ReLU(),
              
            )
        
       
    def forward(self,x,group_idx,fps_idx,N):
        b,c,n,k=x.shape
       
        '''
        group_idx:b,n,k
        new_idx:b,n
        '''
        
        dilated_x=x.permute(0,2,3,1).contiguous().view(b,-1,c)
        group_idx=group_idx.view(b,-1)
        group_x=torch.max(x,dim=-1)[0]
        dilated_x=scatter_max(dilated_x,index=group_idx.long(),dim=1,out=torch.zeros(b,N,c).to(x.device))[0]
        dilated_x=torch.gather(dilated_x,dim=1,index=fps_idx.unsqueeze(-1).repeat(1,1,dilated_x.shape[-1])).permute(0,2,1).contiguous()
        x=torch.cat([group_x,dilated_x],dim=1)
        x=self.convs(x)
        return x
