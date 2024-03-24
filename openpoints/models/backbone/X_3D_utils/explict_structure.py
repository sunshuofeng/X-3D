import torch

import torch.nn as nn
from torch_scatter import scatter_sum,scatter_mean

import math
import torch.nn.functional as F
class PCA(nn.Module):
    def __init__(self,args=None) -> None:
        super().__init__()
        self.outchannel=15
        
    def forward(self,group_xyz,new_xyz,group_idx=None):
        B,N,K,C=group_xyz.shape
 
        X=group_xyz
        X=X.view(B*N,K,C)
        
        u,s,v=torch.linalg.svd(X)
   
        a1=s[:,0]
        a2=s[:,1]
        a3=s[:,2]

        Linearity=(a1-a2)/(a1+1e-10)
        Planarity=(a2-a3)/(a1+1e-10)
        Scattering=a3/(a1+1e-10)

    
        u1=torch.sum(s*torch.abs(v[:,:,0]),dim=-1,keepdim=True)
        
        u2=torch.sum(s*torch.abs(v[:,:,1]),dim=-1,keepdim=True)
        u3=torch.sum(s*torch.abs(v[:,:,2]),dim=-1,keepdim=True)

        direction=torch.cat([u1,u2,u3],dim=-1)
        mean_xyz=new_xyz.view(B*N,3)
        std_xyz=torch.std(group_xyz,dim=2).view(B*N,C)
        norm=v[:,0]

        
       
    
        feature=torch.cat([Linearity.unsqueeze(-1),Planarity.unsqueeze(-1),Scattering.unsqueeze(-1),direction,mean_xyz,std_xyz,norm],dim=-1)
       
      
        return feature





    
class PointHop(nn.Module):
    def __init__(self,args=None) -> None:
        super().__init__()
        self.outchannel=6+24

    

    def forward(self,group_xyz,new_xyz,group_idx=None):
        B,N,K,C=group_xyz.shape
 
        X=group_xyz
        X=X.view(B*N,K,C)


        std_xyz=torch.std(group_xyz,dim=2,keepdim=True).view(B*N,3)
        center=new_xyz.view(B*N,3)
        
        idx=(X[:,:,0]>0).float()*4+(X[:,:,1]>0).float()*2+(X[:,:,2]>0).float()
        current_features=torch.zeros(B*N,8,3).to(group_xyz.device)
        current_features=scatter_mean(X,idx.long(),dim=1,out=current_features).view(B*N,24)

        features=torch.cat([std_xyz,center,current_features],dim=-1)
        return features


class PCA_Pointhop(nn.Module):
    def __init__(self,args=None) -> None:
        super().__init__()
        
    
        self.outchannel=6+24+9

   

    def forward(self,group_xyz,new_xyz,group_idx=None):
        B,N,K,C=group_xyz.shape
 
        X=group_xyz
        X=X.view(B*N,K,C)


        std_xyz=torch.std(group_xyz,dim=2,keepdim=True).view(B*N,3)
        center=new_xyz.view(B*N,3)

        
        idx=(X[:,:,0]>0).float()*4+(X[:,:,1]>0).float()*2+(X[:,:,2]>0).float()

        current_features=torch.zeros(B*N,8,3).to(group_xyz.device)
        current_features=scatter_mean(X,idx.long(),dim=1,out=current_features).view(B*N,24)
        
        u,s,v=torch.linalg.svd(X)

      
        a1=s[:,0]
        a2=s[:,1]
        a3=s[:,2]

        Linearity=(a1-a2)/(a1+1e-10)
        Planarity=(a2-a3)/(a1+1e-10)
        Scattering=a3/(a1+1e-10)

    
        u1=torch.sum(s*torch.abs(v[:,:,0]),dim=-1,keepdim=True)
        
        u2=torch.sum(s*torch.abs(v[:,:,1]),dim=-1,keepdim=True)
        u3=torch.sum(s*torch.abs(v[:,:,2]),dim=-1,keepdim=True)

        direction=torch.cat([u1,u2,u3],dim=-1)
        norm=v[:,:,0]
        features=torch.cat([std_xyz,center,current_features,Linearity.unsqueeze(-1),Planarity.unsqueeze(-1),Scattering.unsqueeze(-1),direction,norm],dim=-1)
        return features

LOCAL_DICT={
    'pca':PCA,
    'pointhop':PointHop,
    'pca_pointhop':PCA_Pointhop,
}