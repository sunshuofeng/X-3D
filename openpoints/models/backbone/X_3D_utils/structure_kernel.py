import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean,scatter_max,scatter_sum




class LLN(nn.Module):
    def __init__(self,inchannel,outchannel) -> None:
        super().__init__()
        self.model=nn.Sequential(nn.Linear(inchannel,outchannel),nn.LayerNorm(outchannel))

    def forward(self,x):
        return self.model(x)
    
import math
class X3D_Model(nn.Module):
    def __init__(self,eschannel,outchannel,convs_pe_mul,args) -> None:
        super().__init__()
        
        #mul position embedding like ptv2 
        self.convs_pe_mul=convs_pe_mul
        hidden_channel=args.hidden_dim
        self.modu=args.modu
    
        
        #denoise module
        self.denoise=args.denoise
    

        if self.denoise!='false':
            self.p_q=nn.Conv1d(3,hidden_channel,kernel_size=1)
            self.p_v=nn.Conv1d(3,hidden_channel,kernel_size=1)

    
        self.convs_pe_mul=convs_pe_mul
        
        self.pre_embed=nn.Sequential(
                nn.Linear(eschannel,hidden_channel),
                nn.LayerNorm(hidden_channel)
            )
        
      
    
        self.filter_size=hidden_channel
        if self.modu!='false':
            outc=outchannel*3
        else:
            outc=outchannel*3+outchannel
        self.outchannel=outchannel
        
        
       
        self.weight_gen=nn.Linear(hidden_channel,outc)
        if self.modu=='false':
            self.norm=nn.Sequential(nn.BatchNorm2d(self.outchannel),nn.ReLU())
      
     



       
       




    def forward(self,fgeo,group_p,group_idx=None,group_f=None):
        b,_,n,k=group_p.shape
        
        
        
        condition=self.pre_embed(fgeo)
       
      
        pe_mul=self.convs_pe_mul(group_p)

       
        B,C_condition=condition.shape

        if self.denoise!='false':
            extent_p=group_p.permute(0,2,1,3).contiguous().view(B,-1,k)
            extent_q=self.p_q(extent_p)
            extent_v=self.p_v(extent_p)
            extent_qk=torch.einsum('bcm,bcn->bmn',extent_q,condition.unsqueeze(-1)).squeeze(-1)
            extent_qk=F.softmax(extent_qk,dim=-1)
            extent_condtion=torch.sum(extent_qk.unsqueeze(1)*extent_v,dim=-1)
            condition=condition+extent_condtion

      
        pe_weight=self.weight_gen(condition)
       
        
        x=group_p.permute(0,2,1,3).contiguous().view(B,-1,k)  
        x=x.view(1,-1,k)
 
            
        filter_size=self.outchannel*3
        if self.modu!='false':
            pe_weight=pe_weight.view(b,n,-1)
            d=pe_weight/math.sqrt(n)
            d=torch.rsqrt(torch.sum(d.pow(2),dim=1,keepdim=True))
            pe_weight=pe_weight*d
            pe_weight=pe_weight.view(B,-1)
            weight=pe_weight[:,:filter_size].contiguous().view(B*self.outchannel,3,1)
            x=F.conv1d(x,weight,groups=B)
            x=x.view(b,n,-1,k).permute(0,2,1,3).contiguous()
            pe_dynamic=x
        else:
            weight=pe_weight[:,:filter_size].contiguous().view(B*self.outchannel,3,1)
            bias=pe_weight[:,-1*self.outchannel:].contiguous().view(-1)
            x=F.conv1d(x,weight,bias,groups=B)
            x=x.view(b,n,-1,k).permute(0,2,1,3).contiguous()
            pe_dynamic=self.norm(x)
       
        group_f=pe_mul*group_f+pe_dynamic
        return group_f
        
       
        
        



     
            
        
    