# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#
# MIXER: open pipe, 160mm length
# NOZZLE: no nozzle installed (TYP: single-taper nozzle, 0.250mm tip)
#   
# =============================================================================

#%% physical constants for mixer and nozzle

PI = 3.141592653589793238462643  # You know what pi is.
L_TOTAL = 0.160  # length of geometry [m]
L_SM = 0.160  # length of pipe [m]
D_MIX = 0.003  # diameter of geometry [m]
A_MIX = PI*D_MIX**2/4  # cross section of geometry [m]
D_IN = 0.003  # inlet diameter of tapered nozzle [m]
D_OUT = 0.00025  # outlet diameter of tapered nozzle [m]
KG = 8  # shear parameter of the pipe [1]
KL_SM = 1  # length parameter of pipe [1]

