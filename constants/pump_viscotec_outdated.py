# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#     
# PUMP: Viscotec (German) progessive-cavity pump
#   
# =============================================================================

# M_ROTOR = 0.001  # mass of rotor [kg]
M_ROTOR = 0.005  # mass of rotor [kg]

D_C = 0.0052  # coupler diameter [m]
L_C = 0.017  # coupler length [m]
T_C = 300  # coupler temperature [K]
RHO_C = 1700  ## coupler density [kg/m^3]
MC_C = 0.60  ## coupler molecular weight of cross-linked strand [kg/mol]
R = 8.314  # gas constant [kg⋅m^2⋅s^−2⋅K^−1⋅mol^−1]

MU_FRIC = 0.4  ## coefficient of friction between rotor and stator [1]
A_RS = 4e-6  # area of contact between rotor and stator [m^2]
R_R = 0.0019  # radius of rotor [m]
S = 0.015  ## depth of contact interference between rotor and stator (normalized) [1]
T_S = 300  # stator temperature [K]
RHO_S = 1400  ## stator density [kg/m^3]
MC_S = 0.70  ## stator molecular weight of the cross-linked strand [kg/mol]

R_SO = 0.0020  # outer radius of the stator [m]
R_SI = 0.0020  # inner radius of the stator [m]
L_S = 0.029  # length of stator [m]

A_CAV = 2.4e-6  # area of inner cavity of pump [m^2]
N_CAV = 2  # number of pump inner cavities at the inlet [1]
PHI = 8  ## angle of attack of pump cavity [deg]

H_S = 0.0015  ## thickness of the stator walls [m]
S_MAX = 0.02  ## maximum allowable change in stator wall thickness (normalized) (TYP:0.1) [1]
R_CAV = 0.0006  # radius of single cavity [m]

W_GAP = 0.1*R_SI # slip gap height [m] (TYP: 0.7)
L_GAP = 0.1*R_SI # slip gap depth [m]  (TYP: 0.7)
K_S = 1.1 # shape adjustment factor for Couette flow [1]

EXTRUSION_RATIO = 2*4.77e-09  # volume of extrusion per stator revolution, FOR THE VISCOTEC PUMP[m^3/rad]
