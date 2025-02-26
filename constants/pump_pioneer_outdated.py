# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#     
# PUMP: Pioneer (Chinese) progessive-cavity pump
#   
# =============================================================================

M_ROTOR = 0.05  # mass of rotor [kg]

D_C = 0.006  # coupler diameter [m]
L_C = 0.05  # coupler length [m]
T_C = 300  # coupler temperature [K]
RHO_C = 1400  # coupler density [kg/m^3]
MC_C = 0.6  # coupler molecular weight of cross-linked strand [kg/mol]
R = 8.314  # gas constant [kg⋅m^2⋅s^−2⋅K^−1⋅mol^−1]

MU_FRIC = 0.5  # coefficient of friction between rotor and stator [1]
A_RS = 5e-07  # area of contact between rotor and stator [m^2]
R_R = 0.002  # radius of rotor [m]
S = 0.01  # depth of contact interference between rotor and stator (normalized) [1]
T_S = 300  # stator temperature [K]
RHO_S = 1600  # stator density [kg/m^3]
MC_S = 0.5  # stator molecular weight of the cross-linked strand [kg/mol]

R_SO = 0.0021  # outer radius of the stator [m]
R_SI = 0.0021  # inner radius of the stator [m]
L_S = 0.060  # length of stator [m]

A_CAV = 3e-5  # area of inner cavity of pump [m^2]
N_CAV = 2  # number of pump inner cavities at the inlet [1]
PHI = 2  # angle of attack of pump cavity [deg]

H_S = 0.005  # thickness of the stator walls [m]
S_MAX = 0.9  # maximum allowable change in stator wall thickness (normalized) [1]
R_CAV = 0.003  # radius of single cavity [m]

W_GAP = 0.3*R_SI # slip gap height [m]
L_GAP = 0.3*R_SI # slip gap depth [mm]
K_S = 1.1 # shape adjustment factor for Couette flow [1]

EXTRUSION_RATIO = 2*1.91e-08  # volume of extrusion per stator revolution, FOR THE CHINESE PUMP[m^3/rad]

