from DEM_Lib import *

EXAMPLE = 2

# Material properties
YM =  1000
PR =  0.3
sig_y0 = 50.

def FlowStressLinear( eps_p_eff ):
    return sig_y0 +  ( YM / 2. ) * eps_p_eff
def FlowStressKinematic( eps_p_eff ):
    return sig_y0 + 0 * eps_p_eff
def HardeningModulusLinear( eps_p_eff ):
    return YM / 2.


UNIFORM = True

print('Plate with hole, cyclic loading')
ISO = True
KINEMATIC = False
FlowStress = FlowStressLinear
ref_file = 'Hole_M3'
rel_tol = np.ones(4) * 2e-5
HardeningModulus = HardeningModulusLinear
BoundingBox = [ 4. , 4. , 1. ] # Size of bounding box
GeometryFile = 'Hole_M3'
disp_schedule = [ 0. , 0.2 , -0.2 , 0.4 , -0.4 ]


base  = './Example' + str(EXAMPLE) + '/'


# Setup domain
domain = setup_domain( base + GeometryFile , BoundingBox )
print('Number of nodes is ', domain['nN'])
print('Number of elements is ', domain['nE'])


# All misc model settings
step_max   = len(disp_schedule) - 1
LBFGS_Iteration = 35
Num_Newton_itr = 1
Settings = [ KINEMATIC , FlowStress , HardeningModulus , disp_schedule , rel_tol , step_max , LBFGS_Iteration , Num_Newton_itr , EXAMPLE , YM , PR , sig_y0 , base , UNIFORM ]

# Hyper parameters
x_var = { 'x_lr' : 0.5 ,
         'neuron' : 100 ,
         'act_func' : 'tanh' }

lr = x_var['x_lr']
H = int(x_var['neuron'])
act_fn = x_var['act_func']
print( 'LR: ' + str(lr) + ', H: ' + str(H) + ', act fn: ' + act_fn )
f = open( base + 'DiffLog','w')
f.close()


# Initial condition, plastic strain and back stress
eps_p = torch.zeros(( domain['nE'] ,3,3))#.double()
PEEQ = torch.zeros(( domain['nE'] ))#.double()
alpha = torch.zeros(( domain['nE'] ,3,3))#.double()

snet = S_Net( 3 , H , 3 , act_fn )
DEM = DeepMixedMethod( [ snet , lr , domain , Settings ] )

tot_sim_time = 0.
prep_times = []
for step in range( 1 , step_max+1 ):
    u_applied = disp_schedule[ step ]

    # Load trained net
    DEM.S_Net.load_state_dict(torch.load( base+'TrainedModel_Step '+str(step)))
    data = DEM.Eval( [eps_p , PEEQ , alpha] , u_applied , step , ref_file )

    # Update
    eps_p , PEEQ , alpha , prep_time , sim_time = data
    prep_times.append( prep_time )
    tot_sim_time += sim_time

tot_sim_time -= np.sum( prep_times[1:] ) # Only really need to do prep once, since mesh is identical in all load steps
print('\n\nShape function gradient prep times = ' , prep_times )
print('DEM simulation time = ' , tot_sim_time )

