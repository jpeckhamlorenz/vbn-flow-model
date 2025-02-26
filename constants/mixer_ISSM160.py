# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#
# MIXER: static mixer, 160mm length, ISSM design
# NOZZLE: no nozzle installed (TYP: single-taper nozzle, 0.250mm tip)
#   
# =============================================================================

#%% physical constants for mixer and nozzle

PI = 3.141592653589793238462643  # You know what pi is.
L_TOTAL = 0.160  # length of geometry [m]
L_SM = 0.160  # length of static mixer [m]
D_MIX = 0.003  # diameter of geometry [m]
A_MIX = PI*D_MIX**2/4  # cross section of geometry [m]
D_IN = 0.003  # inlet diameter of tapered nozzle [m]
D_OUT = 0.002999  # outlet diameter of tapered nozzle [m]
KG = 28  # shear parameter of the static mixer [1] (TYP: 28)
KL_SM = 14.5  # length parameter of static mixer (TYP: 5.5) [1]
L_NOZ = 0.000001  # length of the nozzle (TYP: 0.020) [m]

