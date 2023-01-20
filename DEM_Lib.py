import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
import pyvista as pv
torch.manual_seed(2022)
# torch.cuda.is_available = lambda : False
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    print("CUDA not available, running on CPU")
    dev = torch.device('cpu')
    device_string = 'cpu'


def setup_domain( file , BoundingBox ):
    global CellType , NodePerCell
    nodes , EleConn = [] , []
    readNode = False
    readEle = False
    with open( file + '.inp','r') as fp:
        for cnt, line in enumerate(fp):
            if ( '*Node' in line ):
                readNode = True
                continue
            if ( '*Element' in line ):
                readNode = False
                readEle = True
                continue
            if ('*' in line and readEle ):
                break
            if readNode:
                tmp = line.replace('\n','').split(',')[1:]
                _ = []
                for t in tmp:
                    _.append( float(t) )
                nodes.append(_)
            if readEle:
                tmp = line.replace('\n','').split(',')[1:]
                _ = []
                for t in tmp:
                    _.append( int(t) - 1 ) # Store 0-based indices in element connectivity
                if len(_) == 8:
                    CellType = 12; NodePerCell = 8
                elif len(_) == 4:
                    CellType = 10; NodePerCell = 4
                else:
                    print('Cell type not recognized!')
                    exit()
                EleConn.append(_)
    nodes , EleConn = np.array(nodes) , np.array(EleConn)

    domain           = {}
    domain['Energy'] = torch.from_numpy(nodes)#.float()
    domain['EleConn']     = torch.from_numpy(EleConn).long()
    domain['nE']     = len(EleConn)
    domain['nN']     = len(nodes)
    domain['BB']     = BoundingBox

    if CellType == 12:
        print('Found Hex mesh!')
    else:
        print('Found linear Tet mesh!')

    # # Plot domain
    # cells = np.concatenate( [ np.ones([len(EleConn),1], dtype=np.int32)*NodePerCell , EleConn ] , axis = 1 ).ravel()
    # celltypes = np.empty(domain['nE'], dtype=np.uint8)
    # celltypes[:] = CellType
    # grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    # _ = grid.plot(show_edges=True)
    # exit()

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
    global identity
    lame1 = YM * PR / ( ( 1. + PR ) * ( 1. - 2. * PR ) )
    mu = YM / ( 2. * ( 1. + PR ) )
    trace_e = e[:,0,0] + e[:,1,1] + e[:,2,2]
    return lame1 * torch.einsum( 'ijk,i->ijk' , identity[:len(e),::] , trace_e ) + 2 * mu * e

def Prep_B_physical_Hex( P , conn , nE ):
    P = P.transpose(0,1)

    #        dim
    P_N1 = P[ : , conn[:,0] ]
    P_N2 = P[ : , conn[:,1] ]
    P_N3 = P[ : , conn[:,2] ]
    P_N4 = P[ : , conn[:,3] ]
    P_N5 = P[ : , conn[:,4] ]
    P_N6 = P[ : , conn[:,5] ]
    P_N7 = P[ : , conn[:,6] ]
    P_N8 = P[ : , conn[:,7] ]
    P_N = torch.stack( [ P_N1 , P_N2 , P_N3 , P_N4 , P_N5 , P_N6 , P_N7 , P_N8 ] )#.double()

    x_ , y_ , z_ = 0.,0.,0.
    # Shape grad in natural coords
    B = torch.tensor([[-((y_ - 1)*(z_ - 1))/8, -((x_ - 1)*(z_ - 1))/8, -((x_ - 1)*(y_ - 1))/8],
                        [ ((y_ - 1)*(z_ - 1))/8,  ((x_ + 1)*(z_ - 1))/8,  ((x_ + 1)*(y_ - 1))/8],
                        [-((y_ + 1)*(z_ - 1))/8, -((x_ + 1)*(z_ - 1))/8, -((x_ + 1)*(y_ + 1))/8],
                        [ ((y_ + 1)*(z_ - 1))/8,  ((x_ - 1)*(z_ - 1))/8,  ((x_ - 1)*(y_ + 1))/8],
                        [ ((y_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(z_ + 1))/8,  ((x_ - 1)*(y_ - 1))/8],
                        [-((y_ - 1)*(z_ + 1))/8, -((x_ + 1)*(z_ + 1))/8, -((x_ + 1)*(y_ - 1))/8],
                        [ ((y_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(z_ + 1))/8,  ((x_ + 1)*(y_ + 1))/8],
                        [-((y_ + 1)*(z_ + 1))/8, -((x_ - 1)*(z_ + 1))/8, -((x_ - 1)*(y_ + 1))/8]])#.double()

    # Compute Jacobian
    dPx = torch.einsum( 'ij,iq->qj' , P_N[:,0,:] , B )
    dPy = torch.einsum( 'ij,iq->qj' , P_N[:,1,:] , B )
    dPz = torch.einsum( 'ij,iq->qj' , P_N[:,2,:] , B )
    J = torch.reshape( torch.transpose( torch.cat( (dPx,dPy,dPz) , dim=0 ) , 0 , 1 )  , [nE,3,3] )
    Jinv = torch.linalg.inv( J )
    detJ = torch.linalg.det( J )

    # Convert to physical gradient
    return [ torch.einsum( 'ij,qjk->qik' , B , Jinv ) , detJ ]

def Prep_B_physical_Tet( P , conn , nE ):
    P = P.transpose(0,1)

    #        dim
    P_N1 = P[ : , conn[:,0] ]
    P_N2 = P[ : , conn[:,1] ]
    P_N3 = P[ : , conn[:,2] ]
    P_N4 = P[ : , conn[:,3] ]
    P_N = torch.stack( [ P_N1 , P_N2 , P_N3 , P_N4 ] )#.double()

    g_ , h_ , r_ = 0.,0.,0.
    # Shape grad in natural coords
    B = torch.tensor([[-1., -1., -1.],
                        [ 1.,  0.,  0.],
                        [ 0.,  1.,  0.],
                        [ 0.,  0.,  1.]])#.double()

    # Compute Jacobian
    dPx = torch.einsum( 'ij,iq->qj' , P_N[:,0,:] , B )
    dPy = torch.einsum( 'ij,iq->qj' , P_N[:,1,:] , B )
    dPz = torch.einsum( 'ij,iq->qj' , P_N[:,2,:] , B )
    J = torch.reshape( torch.transpose( torch.cat( (dPx,dPy,dPz) , dim=0 ) , 0 , 1 )  , [nE,3,3] )
    Jinv = torch.linalg.inv( J )
    detJ = torch.linalg.det( J )

    # Convert to physical gradient
    return [ torch.einsum( 'ij,qjk->qik' , B , Jinv ) , detJ ]

def LE_Gauss(U, x, N_element , conn , Ele_info , eps_p , PEEQ , alpha , OUTPUT ):
    B_physical , detJ = Ele_info
    U = U.transpose(0,1)

    U_N = []
    for i in range( len(conn[0,:]) ):
        U_N.append( U[ : , conn[:,i] ] )
    U_N = torch.stack( U_N )#.double()

    # Go through all integration pts
    strainEnergy_at_elem = torch.zeros( N_element )
    wt = 8. if CellType==12 else 1.

    dUx = torch.einsum( 'ij,jil->jl' , U_N[:,0,:] , B_physical )
    dUy = torch.einsum( 'ij,jil->jl' , U_N[:,1,:] , B_physical )
    dUz = torch.einsum( 'ij,jil->jl' , U_N[:,2,:] , B_physical )
    grad_u = torch.reshape( torch.cat( (dUx,dUy,dUz) , dim=-1 ) , [N_element,3,3] )

    # Updated total strain
    strain = 0.5 * ( grad_u + grad_u.permute(0,2,1) )

    # Radial return
    PEEQ_old = PEEQ.clone()
    alpha_old = alpha.clone()
    eps_p_new , PEEQ_new , alpha_new , stress = RadialReturn( strain , eps_p , PEEQ_old , alpha , KINEMATIC )

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
    global identity
    tr_t = t[:,0,0] + t[:,1,1] + t[:,2,2]
    hydro = torch.einsum( 'ijk,i->ijk' , identity[:len(t),::] , tr_t / 3. ) # Hydrostatic part of a tensor
    dev_ = t - hydro # Deviatoric part of a tensor
    return hydro , dev_

def MisesStress( S ):
    return torch.sqrt( 1.5 * torch.einsum( 'ijk,ijk->i' , S , S ) )

def RadialReturn( eps_1 , ep_in , PEEQ_in , alpha_in , KINEMATIC ):
    # Shear modulus
    mu = YM / ( 2. * ( 1. + PR ) )

    # Initialize outputs
    ep_out = ep_in.clone()
    PEEQ_out = PEEQ_in.clone()
    alpha_out = alpha_in.clone()

    # Current flow stress
    flow_stress = FlowStress( PEEQ_out )

    if EXAMPLE == 3:
        flow_stress[ 2168: ] = flow_stress[ 2168: ] + 10.

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
            ep_out[yield_flag] = ep_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , n , d_gamma )
            flow_stress[yield_flag] = FlowStress( PEEQ_out[yield_flag] )    

            # Update full stress tensor
            sig_1[yield_flag] = stressLE( eps_1[yield_flag] - ep_out[yield_flag] )

            # Update back stress tensor
            # # Linear Ziegler hardening, to match Abaqus theory manual
            xi2 = sig_1[yield_flag] - alpha_out[yield_flag]
            norm_xi2 = torch.sqrt( torch.einsum( 'ijk,ijk->i' , xi2 , xi2 ) )
            n2 = torch.einsum( 'ijk,i->ijk' , xi2 , 1. / norm_xi2 )
            # alpha_out[yield_flag] = alpha_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , xi2 , C/sig_y0 * magic_number * d_gamma )


            # Linear Prager hardening
            delta_H = magic_number * C * d_gamma
            alpha_out[yield_flag] = alpha_out[yield_flag] + torch.einsum( 'ijk,i->ijk' , n2 , magic_number * delta_H )

            # # Sanity check
            # hydro , dev_new = decomposition( sig_1[yield_flag] )
            # err = torch.abs( MisesStress( dev_new ) - flow_stress[yield_flag] )
            # print( torch.mean(err).cpu().detach().numpy() )
            # exit()
    return ep_out , PEEQ_out , alpha_out , sig_1

def ConvergenceCheck( arry , rel_tol ):
    num_check = 10

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
        self.S_Net   = model[0]
        self.S_Net   = self.S_Net.to(dev)
        self.lr = model[1]
        self.applied_disp = 0.

        global KINEMATIC , FlowStress , HardeningModulus , disp_schedule , rel_tol , step_max , LBFGS_Iteration , Num_Newton_itr , EXAMPLE , YM , PR , sig_y0 , base , UNIFORM
        KINEMATIC , FlowStress , HardeningModulus , disp_schedule , rel_tol , step_max , LBFGS_Iteration , Num_Newton_itr , EXAMPLE , YM , PR , sig_y0 , base , UNIFORM = model[3]

        # Initialize domain
        self.domain = model[2]
        # Send arrays to device and build indicators
        global nodesEn , EleConn
        nodesEn = self.domain['Energy'].to(dev); nodesEn.requires_grad_(True); nodesEn.retain_grad()
        EleConn = self.domain['EleConn'].to(dev)
        global phix , phiy , phiz
        phix = nodesEn[:, 0] / self.domain['BB'][0]
        phix = phix - torch.min(phix)
        phiy = nodesEn[:, 1] / self.domain['BB'][1]
        phiy = phiy - torch.min(phiy)
        phiz = nodesEn[:, 2] / self.domain['BB'][2]
        phiz = phiz - torch.min(phiz)

        # Store common tensors for reuse
        global identity
        identity = torch.zeros(( self.domain['nE'] , 3, 3)); identity[:,0,0]=1; identity[:,1,1]=1; identity[:,2,2]=1


    def train_model(self , disp_schedule , ref_file ):
        # Get # of parameters
        N_para = 0
        for parameter in self.S_Net.parameters():
            N_para += np.sum( list(parameter.shape) )
        print( 'MLP network has ' , N_para , ' parameters' )
        torch.set_printoptions(precision=8)
        self.S_Net.reset_parameters()

        # Set optimizer
        LBFGS_max_iter  = 200
        optimizerL = torch.optim.LBFGS(self.S_Net.parameters(), lr=self.lr, max_iter=LBFGS_max_iter, line_search_fn='strong_wolfe', tolerance_change=1e-8, tolerance_grad=1e-8)
        LBFGS_loss = {}

        # Initial condition, plastic strain and back stress
        eps_p = torch.zeros(( self.domain['nE'] ,3,3))#.double()
        PEEQ = torch.zeros(( self.domain['nE'] ))#.double()
        alpha = torch.zeros(( self.domain['nE'] ,3,3))#.double()

        # Begin training
        start_time = time.time()
        IO_time = 0.

        # Prep element shape function gradients
        if CellType == 12:
            Ele_info = Prep_B_physical_Hex( nodesEn, EleConn , self.domain['nE'] )
        else:
            Ele_info = Prep_B_physical_Tet( nodesEn, EleConn , self.domain['nE'] )

        all_diff = []
        for step in range(1,step_max+1):
            self.applied_disp = disp_schedule[step]
            print( 'Step ' + str(step) + ' / ' + str(step_max) + ', applied disp = ' + str(self.applied_disp) )
                
            tempL = []
            for epoch in range(LBFGS_Iteration):
                def closure():
                    loss = self.loss_function(step,epoch,nodesEn,self.applied_disp , eps_p , PEEQ , alpha , Ele_info , EleConn , self.domain['nE'] )
                    optimizerL.zero_grad()
                    loss.backward(retain_graph=True)
                    tempL.append(loss.item())
                    return loss
                optimizerL.step(closure)

                # Check convergence
                if ConvergenceCheck( tempL , rel_tol[step-1] ):
                    break
            LBFGS_loss[step] = tempL


            # # Write converged results to file
            start_io_time = time.time()
            u_pred = self.getUP( nodesEn , self.applied_disp )
            Data = LE_Gauss(u_pred, nodesEn, self.domain['nE'] , EleConn , Ele_info , eps_p , PEEQ , alpha , True )
            curr_diff = self.SaveData( self.domain , u_pred , Data , tempL , step , ref_file )
            all_diff.append( curr_diff )

            # Update internal variables
            eps_p = Data[2].to(dev).detach()
            PEEQ = Data[3].to(dev).detach()
            alpha = Data[4].to(dev).detach()
            IO_time += ( time.time() - start_io_time )

            # Save model
            print('Saving trained model')
            torch.save( self.S_Net.state_dict(), base + 'TrainedModel_Step ' + str(step) )

        end_time = time.time()
        print('simulation time = ' + str(end_time - start_time - IO_time) + 's')

        return all_diff

    def getUP(self, nodes , u_applied ):
        uP  = self.S_Net.forward(nodes)#.double()

        if EXAMPLE == 1:
            mag = 0 if UNIFORM else 0.2
            Ux = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 0] + ( phiy + torch.sin( phiy * np.pi * 2 ) * mag ) * u_applied
            Uy = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 1]
            Uz = 0 * uP[:, 2]

            # Ux = phiy * u_applied * 0
            # Uy = phiy * u_applied

        elif EXAMPLE == 2:
            Ux = phix * uP[:, 0] # ux = 0 @ x = 0 
            Uy = phiy * ( 1 - phiy ) * uP[:, 1] + phiy * u_applied # uy = 0 @ y = 0, uy = applied @ y=Ly
            Uz = phiz * uP[:, 2] # uz = 0 @ z = 0 

        elif EXAMPLE == 3:
            Ux = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 0] + phiy * u_applied
            Uy = phix * ( 1 - phix ) * phiy * ( 1 - phiy ) * uP[:, 1]
            Uz = 0 * uP[:, 2]

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        Uz = Uz.reshape(Uz.shape[0], 1)

        u_pred = torch.cat((Ux, Uy, Uz), -1)
        return u_pred


    def loss_function(self,step,epoch,nodesEn,applied_u , eps_p , PEEQ , alpha , Ele_info , EleConn , nE ):
        u_nodesE = self.getUP( nodesEn , applied_u )
        internal = LE_Gauss(u_nodesE, nodesEn, nE , EleConn , Ele_info , eps_p , PEEQ , alpha , False )   
        # print('Step = '+ str(step) + ', Epoch = ' + str( epoch) + ', L = ' + str( internal.item() ) )        
        return internal
    

    def SaveData( self , domain , U , ip_out , LBFGS_loss , step , ref_file ):
        fn = 'Step' + str(step)

        try:
            # Save training loss
            LBFGS_loss_D1 = np.array(LBFGS_loss[1])
            fn_ = base + fn + 'Training_loss.npy'
            np.save( fn_ , LBFGS_loss_D1 )
        except:
            pass

        # Unpack data
        strain_last , stressC_last , strain_plastic_last , PEEQ , alpha = ip_out
        IP_Strain = torch.cat((strain_last[:,0,0].unsqueeze(1),strain_last[:,1,1].unsqueeze(1),strain_last[:,2,2].unsqueeze(1),\
                                  strain_last[:,0,1].unsqueeze(1),strain_last[:,1,2].unsqueeze(1),strain_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Plastic_Strain = torch.cat((strain_plastic_last[:,0,0].unsqueeze(1),strain_plastic_last[:,1,1].unsqueeze(1),strain_plastic_last[:,2,2].unsqueeze(1),\
                                  strain_plastic_last[:,0,1].unsqueeze(1),strain_plastic_last[:,1,2].unsqueeze(1),strain_plastic_last[:,0,2].unsqueeze(1)),axis=1)
        IP_Stress = torch.cat((stressC_last[:,0,0].unsqueeze(1),stressC_last[:,1,1].unsqueeze(1),stressC_last[:,2,2].unsqueeze(1),\
                                  stressC_last[:,0,1].unsqueeze(1),stressC_last[:,1,2].unsqueeze(1),stressC_last[:,0,2].unsqueeze(1)),axis=1)
        stress_vMis = torch.pow(0.5 * (torch.pow((IP_Stress[:,0]-IP_Stress[:,1]), 2) + torch.pow((IP_Stress[:,1]-IP_Stress[:,2]), 2)
                       + torch.pow((IP_Stress[:,2]-IP_Stress[:,0]), 2) + 6 * (torch.pow(IP_Stress[:,3], 2) +
                         torch.pow(IP_Stress[:,4], 2) + torch.pow(IP_Stress[:,5], 2))), 0.5)
        IP_Alpha = torch.cat((alpha[:,0,0].unsqueeze(1),alpha[:,1,1].unsqueeze(1),alpha[:,2,2].unsqueeze(1),\
                                  alpha[:,0,1].unsqueeze(1),alpha[:,1,2].unsqueeze(1),alpha[:,0,2].unsqueeze(1)),axis=1)
        IP_Strain = IP_Strain.cpu().detach().numpy()
        IP_Plastic_Strain = IP_Plastic_Strain.cpu().detach().numpy()
        IP_Stress = IP_Stress.cpu().detach().numpy()
        IP_Alpha = IP_Alpha.cpu().detach().numpy()
        stress_vMis  = stress_vMis.unsqueeze(1).cpu().detach().numpy()
        PEEQ = PEEQ.unsqueeze(1).cpu().detach().numpy()
        U = U.cpu().detach().numpy()

        # Write vtk
        cells = np.concatenate( [ np.ones([self.domain['nE'],1], dtype=np.int32)* NodePerCell , self.domain['EleConn'].numpy() ] , axis = 1 ).ravel()
        celltypes = np.empty(self.domain['nE'], dtype=np.uint8)
        celltypes[:] = CellType
        grid = pv.UnstructuredGrid(cells, celltypes, self.domain['Energy'].numpy() )

        # Nodal data
        names = [ 'Ux' , 'Uy' , 'Uz' ]
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  U[:,idx]

        # Cell data
        names = [ 'E11' , 'E22' , 'E33' , 'E12' , 'E23' , 'E13' , 'Ep11' , 'Ep22' , 'Ep33' , 'Ep12' , 'Ep23' , 'Ep13' ,\
                 'S11' , 'S22' , 'S33' , 'S12' , 'S23' , 'S13' , 'Mises' , 'PEEQ' ,\
                  'A11' , 'A22' , 'A33' , 'A12' , 'A23' , 'A13' ]
        Data = np.concatenate((IP_Strain , IP_Plastic_Strain , IP_Stress , stress_vMis , PEEQ , IP_Alpha ), axis=1)
        for idx , n in enumerate( names ):
            grid.cell_data[ n ] =  Data[:,idx]


        #############################################################################################
        # Abaqus comparison
        step -= 1
        Out1 = np.load( base + ref_file + '_Abaqus.npy' )

        # Displacements
        names = [ 'Ux_Abaqus' , 'Uy_Abaqus' , 'Uz_Abaqus' ]
        for idx , n in enumerate( names ):
            grid.point_data[ n ] =  Out1[idx][step,:self.domain['nN']]
        # Compute difference
        names = [ 'Ux' , 'Uy' , 'Uz' ]
        diff = []
        for idx , n in enumerate( names ):
            FEM = grid.point_data[ n + '_Abaqus' ]
            ML = grid.point_data[ n ]
            grid.point_data[ n + '_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM) ) * 100.
            diff.append( np.mean(grid.point_data[ n + '_diff' ]) )

        # PEEQ at IP
        grid.cell_data[ 'PEEQ_Abaqus' ] =  Out1[3][step,:self.domain['nE']]
        FEM = grid.cell_data[ 'PEEQ_Abaqus' ]
        ML = grid.cell_data[ 'PEEQ' ]
        grid.cell_data[ 'PEEQ_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM[Yield_flag]) + 1e-10 ) * 100.
        diff.append( np.mean(grid.cell_data[ 'PEEQ_diff' ]) )

        # VM at IP
        grid.cell_data[ 'Mises_Abaqus' ] =  Out1[4][step,:self.domain['nE']]
        FEM = grid.cell_data[ 'Mises_Abaqus' ]
        ML = grid.cell_data[ 'Mises' ]
        grid.cell_data[ 'Mises_diff' ] =  np.abs( FEM - ML ) #/ np.mean( np.abs(FEM) + 1e-10 ) * 100.
        diff.append( np.mean(grid.cell_data[ 'Mises_diff' ]) )

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


    def Eval( self , IC , u_applied , step , ref_file ):
        st = time.time()
        # Initial condition, plastic strain and back stress
        eps_p , PEEQ , alpha = IC

        # Prep element shape function gradients
        st2 = time.time()
        if CellType == 12:
            Ele_info = Prep_B_physical_Hex( nodesEn, EleConn , self.domain['nE'] )
        else:
            Ele_info = Prep_B_physical_Tet( nodesEn, EleConn , self.domain['nE'] )
        prep_time = time.time() - st2


        u_pred = self.getUP( nodesEn , u_applied )
        Data = LE_Gauss(u_pred, nodesEn, self.domain['nE'] , EleConn , Ele_info , eps_p , PEEQ , alpha , True )
        sim_time = time.time() - st

        curr_diff = self.SaveData( self.domain , u_pred , Data , None , step , ref_file )

        # Update internal variables
        eps_p = Data[2].to(dev).detach()
        PEEQ = Data[3].to(dev).detach()
        alpha = Data[4].to(dev).detach()
        return [ eps_p , PEEQ , alpha , prep_time , sim_time ]