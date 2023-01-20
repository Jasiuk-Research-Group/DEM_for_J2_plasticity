import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from sklearn.utils import shuffle
import datetime
import h5py
from mpl_toolkits.mplot3d import Axes3D
import torch
print(torch.__version__)
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib as mpl
import numpy.random as npr
import scipy.integrate as sp
from pyevtk.hl import gridToVTK
import pandas as pd 
import numpy.linalg as la
from torch.multiprocessing import Process, Pool
# from NumIntg import *
# import rff
import pyvista as pv
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
torch.manual_seed(2022)
mpl.rcParams['figure.dpi'] = 350

# torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    #torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")


def setup_domain( args ):
    # Physical domain
    Nx , Ny , Nz , DomainTransform = args
    
    # Parent domain
    Length = 2. ; Width = 2. ; Depth = 2.
    x_dom = 0, Length, Nx
    y_dom = 0, Width,  Ny
    z_dom = 0, Depth,  Nz
    # create points
    lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
    lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
    lin_z = np.linspace(z_dom[0], z_dom[1], z_dom[2])
    dom = np.zeros((Nx * Ny * Nz, 3))
    c = 0
    for z in np.nditer(lin_z):
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
            dom[tb:te, 2] = z
    three_D_domain = np.meshgrid(lin_x, lin_y, lin_z)

    # Transform the domains
    dom = DomainTransform( dom )
    ss = three_D_domain[0].shape
    flattened_domain = np.array( [ three_D_domain[i].flatten() for i in range(3) ] ).T
    transformed_flatten = DomainTransform( flattened_domain )
    three_D_domain = [ transformed_flatten[:,i].reshape(ss) for i in range(3) ]

    domain           = {}
    #domain['Energy'] = torch.from_numpy(dom)#.float()
    domain['Energy'] = torch.from_numpy(dom).float()
    domain['3D']     = three_D_domain
    domain['nE']     = ( Nx - 1 ) * ( Ny - 1 ) * ( Nz - 1 )
    return domain

class S_Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out , act_fn):
        super(S_Net, self).__init__()
        self.act_fn = act_fn

        # self.encoding = rff.layers.GaussianEncoding(sigma=0.05, input_size=D_in, encoded_size=H//2)
        # self.encoding = rff.layers.PositionalEncoding(sigma=0.25, m=10)
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, 2*H)
        self.linear3 = torch.nn.Linear(2*H, 4*H)
        self.linear4 = torch.nn.Linear(4*H, 2*H)
        self.linear5 = torch.nn.Linear(2*H, H)
        self.linear6 = torch.nn.Linear(H, D_out)
        
    def forward(self, x ):
        af_mapping = { 'tanh' : torch.tanh ,
                        'relu' : torch.nn.ReLU() ,
                        'rrelu' : torch.nn.RReLU() ,
                        'sigmoid' : torch.sigmoid }
        activation_fn = af_mapping[ self.act_fn ]  
          
        
        # y = self.encoding(x)
        y = activation_fn(self.linear1(x))
        y = activation_fn(self.linear2(y))
        y = activation_fn(self.linear3(y))
        y = activation_fn(self.linear4(y))
        y = activation_fn(self.linear5(y))

        # Output
        y = self.linear6(y)
        return y
    
    def reset_parameters(self):
        for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                    torch.nn.init.normal_(m.bias, mean=0, std=0.1)
                    
def stressLE( e ):
    # global identity
    identity = torch.zeros(( len(e) , 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1

    lame1 = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) )
    mu = YM / ( 2. * ( 1. + PR ) )
    trace_e = e[:,0,0] + e[:,1,1] + e[:,2,2]
    return lame1 * torch.einsum( 'ijk,i->ijk' , identity , trace_e ) + 2 * mu * e

def Prep_B_physical( x , shape ):
    N_element = ( shape[0] - 1 ) * ( shape[1] - 1 ) * ( shape[2] - 1 )
    order = [ 1 ,  shape[-1] , shape[0] , shape[1] ]

    # Fetch nodal coords
    Px = torch.transpose(x[:, 0].reshape( order ), 2, 3)
    Py = torch.transpose(x[:, 1].reshape( order ), 2, 3)
    Pz = torch.transpose(x[:, 2].reshape( order ), 2, 3)
    P = torch.cat( (Px,Py,Pz) , dim=0 )
    #        dim  z      y     x
    P_N1 = P[ : , :-1 , :-1 , :-1 ]
    P_N2 = P[ : , :-1 , :-1 , 1: ]
    P_N3 = P[ : , 1: , :-1 , 1: ]
    P_N4 = P[ : , 1: , :-1 , :-1 ]
    P_N5 = P[ : , :-1 , 1: , :-1 ]
    P_N6 = P[ : , :-1 , 1: , 1: ]
    P_N7 = P[ : , 1: , 1: , 1: ]
    P_N8 = P[ : , 1: , 1: , :-1 ]
    P_N = torch.stack( [ P_N1 , P_N2 , P_N3 , P_N4 , P_N5 , P_N6 , P_N7 , P_N8 ] )#.double()

    x_ , y_ , z_ = 0.,0.,0.
    # Shape grad in natural coords
    B = torch.tensor([[-((y_ - 1)*(z_ - 1))/8, -((x_ - 1)*(z_ - 1))/8, -((x_ - 1)*(y_ - 1))/8],
                [ ((y_ - 1)*(z_ - 1))/8,  ((x_ + 1)*(z_ - 1))/8,  ((x_ + 1)*(y_ - 1))/8],
                [-((y_ - 1)*(z_ + 1))/8, -((x_ + 1)*(z_ + 1))/8, -((x_ + 1)*(y_ - 1))/8],
                [ ((y_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(y_ - 1))/8],
                [ ((y_ + 1)*(z_ - 1))/8,  ((x_ - 1)*(z_ - 1))/8,  ((x_ - 1)*(y_ + 1))/8],
                [-((y_ + 1)*(z_ - 1))/8, -((x_ + 1)*(z_ - 1))/8, -((x_ + 1)*(y_ + 1))/8],
                [ ((y_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(y_ + 1))/8],
                [-((y_ + 1)*(z_ + 1))/8, -((x_ - 1)*(z_ + 1))/8, -((x_ - 1)*(y_ + 1))/8]])#.double()

    # Compute Jacobian
    dPx = torch.einsum( 'ijkl,iq->qjkl' , P_N[:,0,:,:,:] , B )
    dPy = torch.einsum( 'ijkl,iq->qjkl' , P_N[:,1,:,:,:] , B )
    dPz = torch.einsum( 'ijkl,iq->qjkl' , P_N[:,2,:,:,:] , B )
    J = torch.reshape( torch.transpose( torch.flatten( torch.cat( (dPx,dPy,dPz) , dim=0 ) , start_dim=1, end_dim=-1 ) , 0 , 1 )  , [N_element,3,3] )
    Jinv = torch.linalg.inv( J )
    detJ = torch.linalg.det( J )

    # Convert to physical gradient
    return [ torch.einsum( 'ij,qjk->qik' , B , Jinv ) , detJ ]


def DCM( u, x, shape , eps_p , PEEQ , alpha , OUTPUT , indicator ):
    strain        = torch.empty((len(x),3,3))
    stress        = torch.empty((len(x),3,3))

    duxdxyz = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duydxyz = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    duzdxyz = grad(u[:, 2].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
    
    strain[:,0,0] = duxdxyz[:,0]                        # ep_xx
    strain[:,0,1] = 0.5 * (duxdxyz[:,1] + duydxyz[:,0]) # ep_xy
    strain[:,0,2] = 0.5 * (duxdxyz[:,2] + duzdxyz[:,0]) # ep_xz
    strain[:,1,0] = 0.5 * (duxdxyz[:,1] + duydxyz[:,0]) # ep_yx
    strain[:,1,1] = duydxyz[:,1]                        # ep_yy
    strain[:,1,2] = 0.5 * (duydxyz[:,2] + duzdxyz[:,1]) # ep_yz
    strain[:,2,0] = 0.5 * (duxdxyz[:,2] + duzdxyz[:,0]) # ep_zx
    strain[:,2,1] = 0.5 * (duydxyz[:,2] + duzdxyz[:,1]) # ep_zy
    strain[:,2,2] = duzdxyz[:,2]

    # Radial return
    PEEQ_old = PEEQ.clone()
    alpha_old = alpha.clone()
    eps_p_new , PEEQ_new , alpha_new , stress = RadialReturn( strain , eps_p , PEEQ_old , alpha_old , KINEMATIC , indicator )

    dS00dx012 = grad(stress[:,0,0].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS01dx012 = grad(stress[:,0,1].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS02dx012 = grad(stress[:,0,2].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS10dx012 = grad(stress[:,1,0].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS11dx012 = grad(stress[:,1,1].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS12dx012 = grad(stress[:,1,2].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS20dx012 = grad(stress[:,2,0].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS21dx012 = grad(stress[:,2,1].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    dS22dx012 = grad(stress[:,2,2].unsqueeze(1), x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    
    t0 = (dS00dx012[:,0] + dS01dx012[:,1] + dS02dx012[:,2]) 
    t1 = (dS10dx012[:,0] + dS11dx012[:,1] + dS12dx012[:,2]) 
    t2 = (dS20dx012[:,0] + dS21dx012[:,1] + dS22dx012[:,2])  
    
    R_inn = torch.cat((t0,t1,t2),axis = 0) 
    L    = torch.norm(R_inn)
    if not OUTPUT: 
        return L
    else:
        return [ strain , stress , eps_p_new , PEEQ_new , alpha_new ]


def LE_Gauss(u, x, shape , Ele_info , eps_p , PEEQ , alpha , OUTPUT , indicator ):    
    B_physical , detJ = Ele_info
    N_element = ( shape[0] - 1 ) * ( shape[1] - 1 ) * ( shape[2] - 1 )
    order = [ 1 ,  shape[-1] , shape[0] , shape[1] ]

    # Fetch displacements
    Ux = torch.transpose(u[:, 0].reshape( order ), 2, 3)
    Uy = torch.transpose(u[:, 1].reshape( order ), 2, 3)
    Uz = torch.transpose(u[:, 2].reshape( order ), 2, 3)
    U = torch.cat( (Ux,Uy,Uz) , dim=0 )
    #        dim  z      y     x
    U_N1 = U[ : , :-1 , :-1 , :-1 ]
    U_N2 = U[ : , :-1 , :-1 , 1: ]
    U_N3 = U[ : , 1: , :-1 , 1: ]
    U_N4 = U[ : , 1: , :-1 , :-1 ]
    U_N5 = U[ : , :-1 , 1: , :-1 ]
    U_N6 = U[ : , :-1 , 1: , 1: ]
    U_N7 = U[ : , 1: , 1: , 1: ]
    U_N8 = U[ : , 1: , 1: , :-1 ]
    U_N = torch.stack( [ U_N1 , U_N2 , U_N3 , U_N4 , U_N5 , U_N6 , U_N7 , U_N8 ] )#.double()

    # Go through all integration pts
    strainEnergy_at_elem = torch.zeros( N_element )
    wt = 8.
    intpt = torch.tensor([[0.,0.,0.]])

    dUx = torch.einsum( 'ij,jil->jl' , torch.flatten(U_N[:,0,:,:,:], start_dim=1, end_dim=-1 ) , B_physical )
    dUy = torch.einsum( 'ij,jil->jl' , torch.flatten(U_N[:,1,:,:,:], start_dim=1, end_dim=-1 ) , B_physical )
    dUz = torch.einsum( 'ij,jil->jl' , torch.flatten(U_N[:,2,:,:,:], start_dim=1, end_dim=-1 ) , B_physical )
    grad_u = torch.reshape( torch.cat( (dUx,dUy,dUz) , dim=-1 ) , [N_element,3,3] )

    # Updated total strain
    strain = 0.5 * ( grad_u + grad_u.permute(0,2,1) )

    # Radial return
    PEEQ_old = PEEQ.clone()
    alpha_old = alpha.clone()
    eps_p_new , PEEQ_new , alpha_new , stress = RadialReturn( strain , eps_p , PEEQ_old , alpha , KINEMATIC , indicator )

    # Update state
    strain_e = strain - eps_p_new # Updated elastic strain

    # Plastic variables
    delta_eps_tensor = eps_p_new - eps_p
    delta_PEEQ = PEEQ_new - PEEQ_old

    # Compute functional
    K_hard = 0.5 * ( HardeningModulus( PEEQ_new ) + HardeningModulus( PEEQ_old ) )
    if not KINEMATIC:
        SE = 0.5 * torch.einsum( 'ijk,ijk->i' , stress , strain_e ) +\
                0.5 * PEEQ_new * PEEQ_new * K_hard +\
                torch.einsum( 'ijk,ijk->i' , stress , delta_eps_tensor ) -\
                PEEQ_new * K_hard * delta_PEEQ
    else:
        delta_alpha = alpha_new - alpha_old
        SE = 0.5 * torch.einsum( 'ijk,ijk->i' , stress , strain_e ) +\
                0.5 * torch.einsum( 'ijk,ijk->i' , alpha_new , alpha_new ) / K_hard +\
                torch.einsum( 'ijk,ijk->i' , stress , delta_eps_tensor ) -\
                torch.einsum( 'ijk,ijk->i' , alpha_new , delta_alpha ) / K_hard
    strainEnergy_at_elem += SE * wt * detJ   

    if not OUTPUT: 
        return torch.sum( strainEnergy_at_elem )
    else:
        return [ strain , stress , eps_p_new , PEEQ_new , alpha_new ]

def decomposition( t ):
    # global identity
    identity = torch.zeros(( len(t) , 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1

    tr_t = t[:,0,0] + t[:,1,1] + t[:,2,2]
    hydro = torch.einsum( 'ijk,i->ijk' , identity , tr_t / 3. ) # Hydrostatic part of a tensor
    dev_ = t - hydro # Deviatoric part of a tensor
    return hydro , dev_

def MisesStress( S ):
    return torch.sqrt( 1.5 * torch.einsum( 'ijk,ijk->i' , S , S ) )

def RadialReturn( eps_1 , ep_in , PEEQ_in , alpha_in , KINEMATIC , indicator ):
    # Shear modulus
    mu = YM / ( 2. * ( 1. + PR ) )

    # Initialize outputs
    ep_out = ep_in.clone()
    PEEQ_out = PEEQ_in.clone()
    alpha_out = alpha_in.clone()

    # Current flow stress
    flow_stress = FlowStress( PEEQ_out )

    # Elastic guess
    sigma_trial = stressLE( eps_1 - ep_in )
    hydro , deviatoric = decomposition( sigma_trial )
    if not KINEMATIC:
        trial_s_eff = MisesStress( deviatoric )
    else:
        alpha_hydro , alpha_deviatoric = decomposition( alpha_out )
        trial_s_eff = MisesStress( deviatoric - alpha_deviatoric )
    sig_1 = sigma_trial

    # Check for yielding
    yield_flag = ( trial_s_eff >= flow_stress ) # if True, yielding has occurred

    # if EXAMPLE == 6:
    #     yield_flag[ indicator ] = False

    dPEEQ = 0. * trial_s_eff[ yield_flag ] # Initialize as array

    magic_number = np.sqrt(2./3.)
    if len(dPEEQ) > 0: # If at least one point is yielding
        # print('************************************* YIELDING OCCURRED! *********************')
        # Radial return

        if not KINEMATIC:
            # Specializes to linear isotropic hardening
            for itr in range( Num_Newton_itr ):
                H_curr = HardeningModulus( PEEQ_out[yield_flag] ) # Current hardening modulus
                
                # Newton update
                c_pl = ( trial_s_eff[yield_flag] - flow_stress[yield_flag] - 3. * mu * dPEEQ ) / ( 3. * mu + H_curr )
                dPEEQ = dPEEQ + c_pl


                # Scale deviatoric part
                scaler = 1. - 3. * mu * dPEEQ / trial_s_eff[yield_flag]
                dev_new = torch.einsum( 'ijk,i->ijk' , deviatoric[yield_flag] , scaler )

                # Update internal variables
                ep_out[yield_flag] = ep_out[yield_flag] + 1.5 * torch.einsum( 'ijk,i->ijk' , deviatoric[yield_flag] , 1. / trial_s_eff[yield_flag] * c_pl ) 
                PEEQ_out[yield_flag] = PEEQ_out[yield_flag] + c_pl 
                flow_stress[yield_flag] = FlowStress( PEEQ_out[yield_flag] )

                # # Sanity check
                # err = MisesStress( dev_new ) - flow_stress[yield_flag]
                # print( torch.mean(err).detach().numpy() )
            # Update full stress tensor
            sig_1[yield_flag] = hydro[yield_flag] + dev_new
        else:
            # Specializes to linear kinematic hardening
            C = HardeningModulus( PEEQ_out[yield_flag] )

            # Compute return direction
            xi = deviatoric - alpha_deviatoric
            norm_xi = torch.sqrt( torch.einsum( 'ijk,ijk->i' , xi , xi ) )
            n = torch.einsum( 'ijk,i->ijk' , xi , 1. / norm_xi )[yield_flag]

            # Compute plastic multiplier increment
            f_trial = ( norm_xi - magic_number * sig_y0 )[yield_flag]
            d_gamma = f_trial / ( 2*mu + 2.*C / 3. )

            # Update internal variables
            PEEQ_out[yield_flag] = PEEQ_out[yield_flag] + magic_number * d_gamma
            delta_H = magic_number * C * d_gamma
            # alpha_out[yield_flag] = alpha_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , n , magic_number * delta_H )
            ep_out[yield_flag] = ep_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , n , d_gamma )
            flow_stress[yield_flag] = FlowStress( PEEQ_out[yield_flag] )    

            # Update full stress tensor
            sig_1[yield_flag] = stressLE( eps_1[yield_flag] - ep_out[yield_flag] )


            xi2 = sig_1[yield_flag] - alpha_out[yield_flag]
            norm_xi2 = torch.sqrt( torch.einsum( 'ijk,ijk->i' , xi2 , xi2 ) )
            n2 = torch.einsum( 'ijk,i->ijk' , xi2 , 1. / norm_xi2 )

            alpha_out[yield_flag] = alpha_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , n2 , magic_number * delta_H )

            # Sanity check
            # hydro , dev_new = decomposition( sig_1[yield_flag] )
            # err = MisesStress( dev_new ) - flow_stress[yield_flag]
            # print( torch.mean(err).detach().numpy() )
    return ep_out , PEEQ_out , alpha_out , sig_1

def ConvergenceCheck( arry , rel_tol ):
    num_check = 10

    if arry[-1] < -4.:
        print('Solution diverged!!!!!!!')
        return True

    # Run minimum of 2*num_check iterations
    if len( arry ) < 2 * num_check :
        return False

    mean1 = np.mean( arry[ -2*num_check : -num_check ] )
    mean2 = np.mean( arry[ -num_check : ] )

    if np.abs( mean2 ) < 1e-6:
        print('Loss value converged to abs tol of 1e-6' )
        return True     

    if ( np.abs( mean1 - mean2 ) / np.abs( mean2 ) ) < rel_tol:
        print('Loss value converged to rel tol of ' + str(rel_tol) )
        return True
    else:
        return False

class DeepMixedMethod:
    # Instance attributes
    def __init__(self, model):
        self.S_Net   = S_Net(model[0], model[1], model[2] , model[4] )
        self.S_Net   = self.S_Net.to(dev)
        numIntType   = 'simpson'# 'simpson'  'trapezoidal'
        self.lr = model[3]
        self.applied_disp = 0.

    def train_model(self, domain , disp_schedule , super_domain , ref_file ):
        # N_para = 0
        # for parameter in self.S_Net.parameters():
        #     N_para += np.sum( list(parameter.shape) )
        # print( N_para )
        # exit()
        nodesEn = domain['Energy'].to(dev); nodesEn.requires_grad_(True); nodesEn.retain_grad()

        global identity
        identity = torch.zeros(( (Nx-1)*(Ny-1)*(Nz-1) , 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1
        # identity = torch.zeros(( (Nx)*(Ny)*(Nz) , 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1
        torch.set_printoptions(precision=8)
        self.S_Net.reset_parameters()

        if EXAMPLE == 5 :
            # For super resolution
            nodesSuper1 = super_domain[0]['Energy'].to(dev); nodesSuper1.requires_grad_(True); nodesSuper1.retain_grad()
            nodesSuper2 = super_domain[1]['Energy'].to(dev); nodesSuper2.requires_grad_(True); nodesSuper2.retain_grad()
        
        LBFGS_max_iter  = 200
        optimizerL = torch.optim.LBFGS(self.S_Net.parameters(), lr=self.lr, max_iter=LBFGS_max_iter, line_search_fn='strong_wolfe', tolerance_change=1e-8, tolerance_grad=1e-8)
        LBFGS_loss = {}

        # Initial condition, plastic strain and back stress
        if INT_TYPE == 'Gauss':
            eps_pV = torch.zeros(( (Nx-1)*(Ny-1)*(Nz-1) ,3,3))#.double()
            PEEQV = torch.zeros(( (Nx-1)*(Ny-1)*(Nz-1) ))#.double()
            alphaV = torch.zeros(( (Nx-1)*(Ny-1)*(Nz-1) ,3,3))#.double()

            eps_p = torch.zeros(( (Nx)*(Ny)*(Nz) ,3,3))#.double()
            PEEQ = torch.zeros(( (Nx)*(Ny)*(Nz) ))#.double()
            alpha = torch.zeros(( (Nx)*(Ny)*(Nz) ,3,3))#.double()
        else:
            print('Only shape function gradient is implemented!')
            raise NotImplementedError
        # Prep element shape function gradients
        Ele_info = Prep_B_physical( nodesEn, shape )

        # Begin training
        all_diff = []
        for step in range(1,step_max+1):
            self.applied_disp = disp_schedule[step]
            print( 'Step ' + str(step) + ' / ' + str(step_max) + ', applied disp = ' + str(self.applied_disp) )
                
            tempL = []
            for epoch in range(LBFGS_Iteration):
                def closure():
                    loss = self.loss_function(step,epoch,nodesEn,self.applied_disp , eps_p , PEEQ , alpha , Ele_info , eps_pV , PEEQV , alphaV )
                    optimizerL.zero_grad()
                    loss.backward(retain_graph=True)
                    tempL.append(loss.item())
                    return loss
                optimizerL.step(closure)

                # Check convergence
                if ConvergenceCheck( tempL , rel_tol[step-1] ):
                    break
            LBFGS_loss[step] = tempL


            # Write converged results to file
            u_pred = self.getUP( nodesEn , self.applied_disp )

            # internal2 = DCM(u_pred, nodesEn, shape , eps_p , PEEQ , alpha , False )   
            # print('Strong loss = ' + str( internal2.item() ) )        

            Data = LE_Gauss(u_pred, nodesEn, shape , Ele_info , eps_pV , PEEQV , alphaV , True , indicator2 ) 
            curr_diff = self.SaveData( domain , u_pred , Data , tempL , 'Step' + str(step) , step - 1 , [Nx,Ny,Nz] , ref_file )
            all_diff.append( curr_diff )


            Data = DCM(u_pred, nodesEn, shape , eps_p , PEEQ , alpha , True , None )   
            self.SaveData2( domain , u_pred , Data , tempL , 'Step' + str(step) , step - 1 , [Nx,Ny,Nz] , ref_file )  




            # # Update internal variables
            # eps_p = Data[2].to(dev).detach()
            # PEEQ = Data[3].to(dev).detach()
            # alpha = Data[4].to(dev).detach()

        return all_diff

    def getUP(self, nodes , u_applied ):
        uP  = self.S_Net.forward(nodes)#.double()
        phix = nodes[:, 0] / Lx

        if not CYCLIC:
            Ux = phix * uP[:, 0]
            Uy = phix * ( 1 - phix ) * uP[:, 1] + phix * u_applied
        else:
            Ux = phix * ( 1 - phix ) * uP[:, 0] + phix * u_applied
            Uy = phix * uP[:, 1]
        Uz = phix * uP[:, 2]

        if EXAMPLE == 6:
            # Indicator
            phiy = nodes[:, 1] / Lx

            # Constrained shear
            # Ux = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 0] + phiy * u_applied
            Ux = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 0] + ( phiy + torch.sin( phiy * np.pi * 2 ) * 0.2 ) * u_applied
            Uy = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 1]

            # Ux =  ( phiy + torch.sin( phiy * np.pi * 2 ) * 0.2 ) * u_applied
            # Uy = 0 * uP[:, 2]


            Uz = 0 * uP[:, 2]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)

        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred


    def loss_function(self,step,epoch,nodesEn,applied_u , eps_p , PEEQ , alpha , Ele_info , eps_pV , PEEQV , alphaV ):
        u_nodesE = self.getUP( nodesEn , applied_u )
        if INT_TYPE == 'Simpson':
            # internal = LE(u_nodesE, nodesEn, integrationIE, dx, dy, dz, shape , eps_p , alpha )
            print('Autograd may be unstable in this case, not implemented!')
            raise NotImplementedError
        else:
            # internal = LE_Gauss(u_nodesE, nodesEn, shape , Ele_info , eps_pV , PEEQV , alphaV , False )   
            internal = DCM(u_nodesE, nodesEn, shape , eps_p , PEEQ , alpha , False , indicator )   
        # print('Step = '+ str(step) + ', Epoch = ' + str( epoch) + ', L = ' + str( internal.item() ) )        
        # print('Strong loss = ' + str( internal2.item() ) )        
        return internal
    

    def SaveData( self , domain , U , ip_out , LBFGS_loss , fn , step , arg1 , ref_file ):
        Nx,Ny,Nz = arg1

        #######################################################################################################################################
        # Save data
        x_space = np.expand_dims(domain['Energy'][:,0].detach().cpu().numpy(), axis=1)
        y_space = np.expand_dims(domain['Energy'][:,1].detach().cpu().numpy(), axis=1)
        z_space = np.expand_dims(domain['Energy'][:,2].detach().cpu().numpy(), axis=1)
        coordin = np.concatenate((x_space, y_space, z_space), axis=1)

        # Unpack
        strain_last , stressC_last , strain_plastic_last , PEEQ , _ = ip_out

        IP_Strain = torch.cat((strain_last[:,0,0].unsqueeze(1),strain_last[:,1,1].unsqueeze(1),strain_last[:,2,2].unsqueeze(1),\
                                  strain_last[:,0,1].unsqueeze(1),strain_last[:,1,2].unsqueeze(1),strain_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Plastic_Strain = torch.cat((strain_plastic_last[:,0,0].unsqueeze(1),strain_plastic_last[:,1,1].unsqueeze(1),strain_plastic_last[:,2,2].unsqueeze(1),\
                                  strain_plastic_last[:,0,1].unsqueeze(1),strain_plastic_last[:,1,2].unsqueeze(1),strain_plastic_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Stress = torch.cat((stressC_last[:,0,0].unsqueeze(1),stressC_last[:,1,1].unsqueeze(1),stressC_last[:,2,2].unsqueeze(1),\
                                  stressC_last[:,0,1].unsqueeze(1),stressC_last[:,1,2].unsqueeze(1),stressC_last[:,0,2].unsqueeze(1)),axis=1)
        stress_vMis = torch.pow(0.5 * (torch.pow((IP_Stress[:,0]-IP_Stress[:,1]), 2) + torch.pow((IP_Stress[:,1]-IP_Stress[:,2]), 2)
                       + torch.pow((IP_Stress[:,2]-IP_Stress[:,0]), 2) + 6 * (torch.pow(IP_Stress[:,3], 2) +
                         torch.pow(IP_Stress[:,4], 2) + torch.pow(IP_Stress[:,5], 2))), 0.5)
        IP_Strain = IP_Strain.cpu().detach().numpy()
        IP_Plastic_Strain = IP_Plastic_Strain.cpu().detach().numpy()
        IP_Stress = IP_Stress.cpu().detach().numpy()
        stress_vMis  = stress_vMis.unsqueeze(1).cpu().detach().numpy()
        PEEQ = PEEQ.unsqueeze(1).cpu().detach().numpy()
        U = U.cpu().detach().numpy()


        # Save as npy
        # np.save( base + fn + '.npy', np.array([ coordin , U , IP_Stress , IP_Strain , IP_Plastic_Strain , stress_vMis , PEEQ ],dtype=object) )
        # LBFGS_loss_D1 = np.array(LBFGS_loss[1])
        # fn_ = base + fn + 'Training_loss.npy'
        # np.save( fn_ , LBFGS_loss_D1 )


        #######################################################################################################################################
        # Write vtk
        def FormatMe( v , S ):
            return np.swapaxes( np.swapaxes( v.reshape(S) , 0 , 1 ) , 1 , 2 ).flatten('F')

        grid = pv.StructuredGrid( domain['3D'][0] , domain['3D'][1] , domain['3D'][2] )

        # Nodal data
        names = [ 'Ux' , 'Uy' , 'Uz' ]
        # S = [Nz,Nx,Ny]
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  U[:,idx]

        # Cell data
        names = [ 'E11' , 'E22' , 'E33' , 'E12' , 'E23' , 'E13' , 'Ep11' , 'Ep22' , 'Ep33' , 'Ep12' , 'Ep23' , 'Ep13' , 'S11' , 'S22' , 'S33' , 'S12' , 'S23' , 'S13' , 'SvM' , 'PEEQ' ]
        Data = np.concatenate((IP_Strain , IP_Plastic_Strain , IP_Stress , stress_vMis , PEEQ ), axis=1)
        S = [Nz-1,Ny-1,Nx-1]
        for idx , n in enumerate( names ):
            grid.cell_data[ n ] =  FormatMe( Data[:,idx] , S )

        # #############################################################################################
        # Abaqus comparison
        Out1 = np.load( base + ref_file + '_disp.npy' )
        Out2 = np.load( base + ref_file + '_PEEQ.npy' )

        # Displacements
        names = [ 'Ux_ABQ' , 'Uy_ABQ' , 'Uz_ABQ' ]
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  np.swapaxes( Out1[idx][step,:,:,:] , 0 , 1 ).flatten('F')
        # Compute difference
        names = [ 'Ux' , 'Uy' , 'Uz' ]
        diff = []
        for idx , n in enumerate( names ):
            FEM = grid.point_data[ n + '_ABQ' ]
            ML = grid.point_data[ n ]
            grid.point_data[ n + '_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM) ) * 100.
            diff.append( np.mean(grid.point_data[ n + '_diff' ]) )

        # PEEQ at IP
        grid.cell_data[ 'PEEQ_ABQ' ] =  np.swapaxes( Out2[0][step,:,:,:] , 0 , 1 ).flatten('F')

        # Check yield
        # Yield_flag = ( grid.cell_data[ 'PEEQ_ABQ' ] > 0.002 )

        FEM = grid.cell_data[ 'PEEQ_ABQ' ]
        ML = grid.cell_data[ 'PEEQ' ]
        grid.cell_data[ 'PEEQ_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM[Yield_flag]) + 1e-10 ) * 100.
        diff.append( np.mean(grid.cell_data[ 'PEEQ_diff' ]) )

        # # VM at IP
        grid.cell_data[ 'SvM_ABQ' ] =  np.swapaxes( Out2[1][step,:,:,:] , 0 , 1 ).flatten('F')
        FEM = grid.cell_data[ 'SvM_ABQ' ]
        ML = grid.cell_data[ 'SvM' ]
        grid.cell_data[ 'SvM_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM) + 1e-10 ) * 100.
        diff.append( np.mean(grid.cell_data[ 'SvM_diff' ]) )

        # Write
        grid.save( base + fn + "Results.vtk")

        # Check mean error
        f = open( base + 'DiffLog','a')
        f.write('Step ' + str(step+1) + '\n' )
        print('Step ' + str(step+1) )
        f.write( 'Component-wise error: \n' )
        print( 'Component-wise error: ' )
        for dd , tit in zip( diff , ['Ux','Uy','Uz','PEEQ','SvM'] ):
            f.write('Mean error in ' + tit + ' = ' + str(dd) + ' \n' )
            print('Mean error in ' + tit + ' = ' + str(dd) + ' ' )
        f.close()

        return diff




    def SaveData2( self , domain , U , ip_out , LBFGS_loss , fn , step , arg1 , ref_file ):
        Nx,Ny,Nz = arg1

        #######################################################################################################################################
        # Save data
        x_space = np.expand_dims(domain['Energy'][:,0].detach().cpu().numpy(), axis=1)
        y_space = np.expand_dims(domain['Energy'][:,1].detach().cpu().numpy(), axis=1)
        z_space = np.expand_dims(domain['Energy'][:,2].detach().cpu().numpy(), axis=1)
        coordin = np.concatenate((x_space, y_space, z_space), axis=1)

        # Unpack
        strain_last , stressC_last , strain_plastic_last , PEEQ , _ = ip_out

        IP_Strain = torch.cat((strain_last[:,0,0].unsqueeze(1),strain_last[:,1,1].unsqueeze(1),strain_last[:,2,2].unsqueeze(1),\
                                  strain_last[:,0,1].unsqueeze(1),strain_last[:,1,2].unsqueeze(1),strain_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Plastic_Strain = torch.cat((strain_plastic_last[:,0,0].unsqueeze(1),strain_plastic_last[:,1,1].unsqueeze(1),strain_plastic_last[:,2,2].unsqueeze(1),\
                                  strain_plastic_last[:,0,1].unsqueeze(1),strain_plastic_last[:,1,2].unsqueeze(1),strain_plastic_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Stress = torch.cat((stressC_last[:,0,0].unsqueeze(1),stressC_last[:,1,1].unsqueeze(1),stressC_last[:,2,2].unsqueeze(1),\
                                  stressC_last[:,0,1].unsqueeze(1),stressC_last[:,1,2].unsqueeze(1),stressC_last[:,0,2].unsqueeze(1)),axis=1)
        stress_vMis = torch.pow(0.5 * (torch.pow((IP_Stress[:,0]-IP_Stress[:,1]), 2) + torch.pow((IP_Stress[:,1]-IP_Stress[:,2]), 2)
                       + torch.pow((IP_Stress[:,2]-IP_Stress[:,0]), 2) + 6 * (torch.pow(IP_Stress[:,3], 2) +
                         torch.pow(IP_Stress[:,4], 2) + torch.pow(IP_Stress[:,5], 2))), 0.5)
        IP_Strain = IP_Strain.cpu().detach().numpy()
        IP_Plastic_Strain = IP_Plastic_Strain.cpu().detach().numpy()
        IP_Stress = IP_Stress.cpu().detach().numpy()
        stress_vMis  = stress_vMis.unsqueeze(1).cpu().detach().numpy()
        PEEQ = PEEQ.unsqueeze(1).cpu().detach().numpy()
        U = U.cpu().detach().numpy()


        grid = pv.StructuredGrid( domain['3D'][0] , domain['3D'][1] , domain['3D'][2] )

        # Nodal data
        names = [ 'Ux' , 'Uy' , 'Uz' ]
        # S = [Nz,Nx,Ny]
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  U[:,idx]

        # Cell data
        names = [ 'E11' , 'E22' , 'E33' , 'E12' , 'E23' , 'E13' , 'Ep11' , 'Ep22' , 'Ep33' , 'Ep12' , 'Ep23' , 'Ep13' , 'S11' , 'S22' , 'S33' , 'S12' , 'S23' , 'S13' , 'SvM' , 'PEEQ' ]
        Data = np.concatenate((IP_Strain , IP_Plastic_Strain , IP_Stress , stress_vMis , PEEQ ), axis=1)
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  Data[:,idx]

        # Write
        grid.save( base + fn + "Results_Nodal.vtk")

        return 0



EXAMPLE = 6
INT_TYPE = 'Gauss'
NGP = 1


print('Using ' + INT_TYPE + ' integration with ' + str(NGP) + ' integration points' )
base  = './Example' + str(EXAMPLE) + '/'
# ------------------------------ network settings ---------------------------------------------------
D_in  = 3
D_out = 3

# ----------------------------- define structural parameters ----------------------------------------
# Number of nodes
Nx = 60 + 1 ; Ny = 15 + 1 ; Nz = 15 + 1

# Material properties
YM =  1000
PR =  0.3
sig_y0 = 50.


def FlowStressLinear( eps_p_eff ):
    return sig_y0 +  ( YM / 2. ) * eps_p_eff
def HardeningModulusLinear( eps_p_eff ):
    return YM / 2.
def FlowStressPerfectPlastic( eps_p_eff ):
    return sig_y0 + 0 * eps_p_eff
def HardeningModulusPerfectPlastic( eps_p_eff ):
    return 0 * eps_p_eff
def FlowStressKinematic( eps_p_eff ):
    return sig_y0 + 0 * eps_p_eff
def HardeningModulusKinematic( eps_p_eff ):
    return YM / 2.


# Definition of domain
# Rectangular beam, 4x1x1 m
def DomainTransform1( dom ):
    dom[:,0] *= 2. # X
    dom[:,1] *= 0.5 # Y
    dom[:,2] *= 0.5 # Z
    return dom

# Cook's membrane, 48x60x1 m
def DomainTransform2( dom ):
    # Convert to natural coords, [-1,1]
    xi , eta = dom[:,0] - 1. , dom[:,1] - 1.

    # Physical domain
    #          N1   N2    N3    N4
    X_phys = [ 0. , 48. , 48. , 0. ]
    Y_phys = [ 0. , 44. , 60. , 44. ]

    # Shape functions
    N1 = 0.25 * ( 1. - xi ) * ( 1. - eta )
    N2 = 0.25 * ( 1. + xi ) * ( 1. - eta ) 
    N3 = 0.25 * ( 1. + xi ) * ( 1. + eta ) 
    N4 = 0.25 * ( 1. - xi ) * ( 1. + eta ) 

    # Map
    dom[:,0] = N1 * X_phys[0] + N2 * X_phys[1] + N3 * X_phys[2] + N4 * X_phys[3] 
    dom[:,1] = N1 * Y_phys[0] + N2 * Y_phys[1] + N3 * Y_phys[2] + N4 * Y_phys[3] 

    dom[:,2] *= 0.5 # Z
    return dom

# Square plate, 4x4x1 m
def DomainTransform3( dom ):
    dom[:,0] *= 2. # X
    dom[:,1] *= 2 # Y
    dom[:,2] *= 0.5 # Z
    return dom


# Setup examples
if EXAMPLE == 1:
    print('Linear isotropic hardening, monotonic loading')
    KINEMATIC = False
    CYCLIC = False
    Num_Newton_itr = 1
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear
    Lx = 4.
    DomainTransform = DomainTransform1
    # disp_schedule = [ 0. , -2 , -4. ]
    # rel_tol = [ 5e-5 , 5e-5 ]
    # ref_file = 'LH_u4_2'
    disp_schedule = [ 0. , -4. ]
    rel_tol = [ 5e-5 ]
    ref_file = 'LH_u4_1'


elif EXAMPLE == 2:
    print('Perfect plasticity, monotonic loading')
    KINEMATIC = False
    CYCLIC = False
    Num_Newton_itr = 1
    FlowStress = FlowStressPerfectPlastic
    HardeningModulus = HardeningModulusPerfectPlastic
    Lx = 4.
    DomainTransform = DomainTransform1
    # disp_schedule = [ 0. , -0.75 , -1.5 ]
    # rel_tol = [ 2e-5 , 1e-4 ]
    # ref_file = 'PP_u15_2'
    # disp_schedule = [ 0. , -1.5 ]
    # rel_tol = [ 1e-4 ]
    # ref_file = 'PP_u15_1'

    disp_schedule = [ 0. , -2 , -4. ]
    rel_tol = [ 1e-5 , 1e-5 ]
    ref_file = 'NH_u4_2'

elif EXAMPLE == 3:
    print('Linear isotropic hardening, cyclic loading')
    KINEMATIC = False
    CYCLIC = True
    Num_Newton_itr = 1
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear
    Lx = 4.
    DomainTransform = DomainTransform1
    disp_schedule = [ 0. , 1. , 0.1 , -1. , -0.1 ]
    rel_tol = [ 2e-5 , 1e-6 , 2e-5 , 1e-6 ]
    ref_file = 'CL_S4_iso'

elif EXAMPLE == 4:
    print('Linear kinematic hardening, cyclic loading')
    KINEMATIC = True
    CYCLIC = True
    Num_Newton_itr = 1
    FlowStress = FlowStressKinematic
    HardeningModulus = HardeningModulusKinematic
    Lx = 4.
    DomainTransform = DomainTransform1
    disp_schedule = [ 0. , 1. , 0.1 , -1. , -0.1 ]
    rel_tol = [ 1e-4 , 1e-6 , 1e-4 , 2e-5 ]
    ref_file = 'CL_S4'

elif EXAMPLE == 5:
    print('Cooks membrane with DEM inference' )
    Nx = 48 + 1 ; Ny = 30 + 1 ; Nz = 2 + 1
    # Nx = 96 + 1 ; Ny = 60 + 1 ; Nz = 2 + 1


    KINEMATIC = False
    CYCLIC = False
    Num_Newton_itr = 1
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear
    Lx = 48.
    DomainTransform = DomainTransform2

    disp_schedule = [ 0. , 10. ]
    rel_tol = [ 5e-6 ]
    ref_file = 'Cook_LH_u10_M1'

elif EXAMPLE == 6:
    print('Constrained shear with inclusion' )
    Nx = 100 + 1 ; Ny = 100 + 1 ; Nz = 1 + 1

    KINEMATIC = True
    CYCLIC = False
    Num_Newton_itr = 1
    FlowStress = FlowStressKinematic
    HardeningModulus = HardeningModulusLinear
    Lx = 4.
    DomainTransform = DomainTransform3

    # Make indicator function
    global indicator
    # indicator = np.zeros([Nx-1,Ny-1])
    indicator = np.zeros([Nx,Ny])
    def Rect( i , j , a , b , indicator , val ):
        indicator[ j:j+b , i:i+a ] = val
    Rect( 50 - 20 , 50 - 28 , 40 , 56 , indicator , -1 )
    Rect( 50 - 20 , 50 - 28 + 14 , 9 , 27 , indicator , 0 )
    Rect( 50 - 20 + 31 , 50 - 28 + 14 , 9 , 27 , indicator , 0 )

    # plt.matshow( np.flipud(indicator) )
    # plt.show()

    tmp = torch.from_numpy( indicator.flatten() )
    indicator = torch.gt( tmp , -0.5 )
    indicator = torch.cat([indicator,indicator],axis=0)


    indicator2 = np.zeros([Nx-1,Ny-1])
    def Rect( i , j , a , b , indicator , val ):
        indicator[ j:j+b , i:i+a ] = val
    Rect( 50 - 20 , 50 - 28 , 40 , 56 , indicator2 , -1 )
    Rect( 50 - 20 , 50 - 28 + 14 , 9 , 27 , indicator2 , 0 )
    Rect( 50 - 20 + 31 , 50 - 28 + 14 , 9 , 27 , indicator2 , 0 )

    # plt.matshow( np.flipud(indicator) )
    # plt.show()
    tmp2 = torch.from_numpy( indicator2.flatten() )
    indicator2 = torch.gt( tmp2 , -0.5 )

    disp_schedule = [ 0. , .5 ]
    rel_tol = [ 1e-6 ]
    ref_file = 'shearWave'


shape = [Nx, Ny, Nz]

domain = setup_domain( [ Nx , Ny , Nz , DomainTransform ] )
print('# of nodes is ', len(domain['Energy']))

# Super resolution
if EXAMPLE == 5:
    Nxs1 = 96 + 1 ; Nys1 = 60 + 1 ; Nzs1 = 2 + 1
    super_domain1 = setup_domain( [ Nxs1 , Nys1 , Nzs1 , DomainTransform ] )
    Nxs2 = 192 + 1 ; Nys2 = 120 + 1 ; Nzs2 = 2 + 1
    super_domain2 = setup_domain( [ Nxs2 , Nys2 , Nzs2 , DomainTransform ] )
else:
    super_domain1 = None
    super_domain2 = None



# Loading
step_max   = len(disp_schedule) - 1

# Training
LBFGS_Iteration = 35

# Hyper parameters
x_var = { 'x_lr' : 0.5 ,
         'neuron' : 100 ,
         'act_func' : 'tanh' }

def Obj( x_var ):
    lr = x_var['x_lr']
    H = int(x_var['neuron'])
    act_fn = x_var['act_func']
    print( 'LR: ' + str(lr) + ', H: ' + str(H) + ', act fn: ' + act_fn )
    f = open( base + 'DiffLog','w')
    f.close()


    dcm = DeepMixedMethod([D_in, H, D_out, lr , act_fn])
    start_time = time.time()
    all_diff = dcm.train_model(domain , disp_schedule , [ super_domain1 , super_domain2 ] , ref_file )
    end_time = time.time()
    print('simulation time = ' + str(end_time - start_time) + 's')

    fn_ = base + 'AllDiff.npy'
    np.save( fn_ , all_diff )

    return 0

Obj( x_var )
