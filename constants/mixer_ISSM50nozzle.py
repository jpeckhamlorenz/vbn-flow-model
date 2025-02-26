# =============================================================================
# 
# THIS MODULE DEFINES PROJECT-LEVEL CONSTANTS FOR:
#
# MIXER: static mixer, 50mm length, ISSM design
# NOZZLE: tapered nozzle installed (TYP: single-taper nozzle, 0.250mm tip)
#   
# =============================================================================

#%% physical constants for mixer and nozzle

PI = 3.141592653589793238462643  # You know what pi is.
L_TOTAL = 0.050  # length of geometry [m]  (TYP: 0.070)
L_SM = 0.050  # length of static mixer [m]  (TYP: 0.050)
D_MIX = 0.003  # diameter of geometry [m]
A_MIX = PI*D_MIX**2/4  # cross section of geometry [m]
D_IN = 0.003  # inlet diameter of tapered nozzle [m]
D_OUT = 0.00025  # outlet diameter of tapered nozzle [m]
KG = 28  # shear parameter of the static mixer [1]
KL_SM = 5.5  # length parameter of static mixer (TYP: 13.5) [1]
L_NOZ = 0.020  # length of the nozzle (TYP: 0.020) [m]
