from DEM_Lib import *

EXAMPLE = 3

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


# Setup examples
UNIFORM = True
if EXAMPLE == 1:
    print('Constrained shear')
    KINEMATIC = False
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear

    if UNIFORM: # Get stress-strain curve
        ref_file = 'shear10_iso'

        KINEMATIC = True
        FlowStress = FlowStressKinematic
        ref_file = 'shear10_kine'
        
        # disp_schedule = np.linspace( 0. , 1. , 11 )
        disp_schedule = [ 0. , 1./3. , 2./3. , 1. , 2./3. , 1./3. , 0. , -1./3. , -2./3. , -1. , -2./3. , -1./3. , 0. ]
        rel_tol = np.ones( 13 ) * 1e-6

    else: # Compare with DCM
        ref_file = 'shearWave'

        KINEMATIC = True
        FlowStress = FlowStressKinematic
        ref_file = 'ShearWave_kine'


        disp_schedule = [ 0. , 0.5 ]
        rel_tol = [ 1e-6 ]

    BoundingBox = [ 4. , 4. , 1. ] # Size of bounding box
    GeometryFile = 'Shear'


elif EXAMPLE == 2:
    print('Plate with hole, cyclic loading')
    ISO = True
    if ISO:
        KINEMATIC = False
        FlowStress = FlowStressLinear
        ref_file = 'Hole'
        rel_tol = np.ones(4) * 2e-5
    else:
        KINEMATIC = True
        FlowStress = FlowStressKinematic
        ref_file = 'HoleKine'
        rel_tol = np.ones(4) * 2e-5


    HardeningModulus = HardeningModulusLinear
    BoundingBox = [ 4. , 4. , 1. ] # Size of bounding box
    GeometryFile = 'Hole'
    disp_schedule = [ 0. , 0.2 , -0.2 , 0.4 , -0.4 ]


elif EXAMPLE == 3:
    print('Bimaterial plate')
    KINEMATIC = False
    FlowStress = FlowStressLinear
    HardeningModulus = HardeningModulusLinear
    ref_file = 'BiMat'
    disp_schedule = [ 0. , 0.5 ]
    rel_tol = np.ones( 1 ) * 1.e-6
    BoundingBox = [ 4. , 4. , 1. ] # Size of bounding box
    GeometryFile = 'BiMat'



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

# Begin training
snet = S_Net( 3 , H , 3 , act_fn )
DEM = DeepMixedMethod( [ snet , lr , domain , Settings ] )
all_diff = DEM.train_model( disp_schedule , ref_file )
fn_ = base + 'AllDiff.npy'
np.save( fn_ , all_diff )