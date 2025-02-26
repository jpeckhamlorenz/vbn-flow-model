# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#     
# PUMP: Pioneer (Chinese) progessive-cavity pump
#   
# =============================================================================

# M_ROTOR = 0.005  # mass of rotor [kg]
M_ROTOR = 0.015  # mass of rotor [kg]

D_C = 0.007  # coupler diameter [m]
L_C = 0.035  # coupler length [m]
T_C = 290  # coupler temperature [K]
RHO_C = 1200  ## coupler density [kg/m^3]
MC_C = 0.8  ## coupler molecular weight of cross-linked strand [kg/mol]
R = 8.314  # gas constant [kg⋅m^2⋅s^−2⋅K^−1⋅mol^−1]

MU_FRIC = 0.6  ## coefficient of friction between rotor and stator [1]
A_RS = 9e-6  # area of contact between rotor and stator [m^2]
R_R = 0.0026  # radius of rotor [m]
S = 0.02  ## depth of contact interference between rotor and stator (normalized) [1]
T_S = 290  # stator temperature [K]
RHO_S = 1700  ## stator density [kg/m^3]
MC_S = 0.4  ## stator molecular weight of the cross-linked strand [kg/mol]

R_SO = 0.0027  # radius of the stator [m]
R_SI = 0.0027  # inner radius of the stator [m]
L_S = 0.045  # length of stator [m]

A_CAV = 7e-6  # area of inner cavity of pump [m^2]
N_CAV = 2  # number of pump inner cavities at the inlet [1]
PHI = 35  ## angle of attack of pump cavity [deg]

H_S = 0.002  # thickness of the stator walls [m]
S_MAX = 0.9  ## maximum allowable change in stator wall thickness (normalized) [1]
R_CAV = 0.001  # radius of single cavity [m]

W_GAP = 0.2*R_SI # slip gap height [m]
L_GAP = 0.2*R_SI # slip gap depth [mm]
K_S = 1.1 # shape adjustment factor for Couette flow [1]

EXTRUSION_RATIO = 2*1.91e-08  # volume of extrusion per stator revolution, FOR THE CHINESE PUMP[m^3/rad]



