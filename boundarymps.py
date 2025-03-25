import torch,math
import numpy as np
from scipy.linalg import sqrtm

def get_lnZ(L,beta=0.44068679350977147,chi=16,mydtype=torch.float64,mydevice=torch.device('cpu')):
    assert(L%2==0)
    B=torch.tensor(sqrtm(np.array([[np.exp(beta),np.exp(-beta)],[np.exp(-beta),np.exp(beta)]])),dtype=mydtype,device=mydevice) # the Boltzmann matrix
    A2=B@B #A2=torch.einsum("ij,ab,jb->ia",[B,B,I2])
    A3=torch.einsum("i,j,k->ijk",B[:,0],B[:,0],B[:,0])+torch.einsum("i,j,k->ijk",B[:,1],B[:,1],B[:,1]) #A3=torch.einsum("ij,ab,cd,jbd->iac",[B,B,B,I3]) 
    A4=torch.einsum("i,j,k,l->ijkl",B[:,0],B[:,0],B[:,0],B[:,0])+torch.einsum("i,j,k,l->ijkl",B[:,1],B[:,1],B[:,1],B[:,1]) # A4=torch.einsum("ij,ab,cd,xy,jbdy->iacx",[B,B,B,B,I4])
    tensors=[]
    tensors.append( [A2[None,:,:] if i==0 else( A2[:,:,None] if i==L-1 else A3) for i in range(L)] )
    for j in range(1,L>>1):
        tensors.append( [ A3[None,:,:,:] if i==0 else( A3[:,:,None,:] if i==L-1 else A4)  for i in range(L) ] )
    lnZ=0 # log of partition function
    for head in range((L>>1)-1): # mps on the boundary is eating the next mpo, for L/2-1 times
        [res,tensors[head+1]] = compress( eat(tensors[head][:],tensors[head+1][:]) , chi)
        lnZ += res
    return 2*lnZ

def eat(mps,mpo):
    return [ torch.einsum("ijk,abcj->iabkc",mps[i],mpo[i]).contiguous().view(mps[i].shape[0]*mpo[i].shape[0],2,-1) for i in range(len(mps))]     

def compress(mps,chi):
    residual=0
    for i in range(len(mps)-1): # From left to right, sweep once doing qr decompositions
        Q,R=torch.qr(mps[i].contiguous().view(mps[i].shape[0]*2,-1))
        mps[i] = Q.contiguous().view(mps[i].shape[0],2,-1)
        mps[i+1] = torch.einsum("ij,jab->iab",[R,mps[i+1]])
    for i in range(len(mps)-1,0,-1): # From right to left, sweep onece using svd on the tensor merged from two consecutive tensors.
        [U,s,V]=torch.svd( torch.einsum("ijk,kab->ijab",mps[i-1],mps[i]).view(mps[i-1].shape[0]*2,mps[i].shape[2]*2) )
        mps[i] = V[:,:chi].t().contiguous().view(-1,2,mps[i].shape[2])
        mps[i-1] = (U[:,:chi]@torch.diag(s[:chi])).contiguous().view(mps[i-1].shape[0],2,-1)
        tnorm=mps[i-1].norm()
        mps[i-1] /= tnorm
        residual += math.log(tnorm)
    return residual,mps

import kacward 
L=16
beta_c=0.44068679350977147
chi=16
print("L=",L," chi=",chi)
lnZ=get_lnZ(L=L,beta=beta_c,chi=chi);print("lnZ_TN=",lnZ/L**2)
lnZ_exact=kacward.lnZ_2d_ferro_Ising(L,beta_c);print("lnZ_Exact=",lnZ_exact/L**2)
print("|lnZ-lnZ_exact|=%.2g"%(abs(lnZ-lnZ_exact)/L**2))


