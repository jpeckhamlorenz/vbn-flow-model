# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#     
# FLUID: Dow 121 silicone
#   
# =============================================================================

WAVESPEED = 880  # acoustic wavespeed, dx/dt [m/s]
RHO = 1295  # fluid density [kg/m^3]
K_INDEX = 650  # flow consistency index [Pa*s^n]
MU = 20  # dynamic viscosity (unused in non-newtonian model) [Pa*s]
MU_INF = 5000  # approximated dynamic viscosity when fluid is at rest [Pa*s]
N_INDEX = 0.6  # flow behavior index [1]
Z_SHIFT = (MU_INF/K_INDEX)**(1/(N_INDEX-1))
