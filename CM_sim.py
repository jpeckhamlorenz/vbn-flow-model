# import re
# import itertools
import numpy as np
# import pandas as pd
# import glob
# import matplotlib
import matplotlib.pyplot as plt
# import platform
# import skimage
# import imageio
# import av
# import imghdr
# from scipy.signal import medfilt, savgol_filter as sg_filter
from scipy.optimize import curve_fit
# import cv2 as cv
# import torch
# import torchvision
# from segment_anything import sam_model_registry, SamPredictor
# import gc
from time import perf_counter
# import os.path
from copy import deepcopy

from constants import fluid_DOW121 as fld
# from constants import mixer_ISSM50nozzle as mix
# from constants import pump_viscotec_outdated as pmp

from inputs import corner_naive as gen

#%% initialization 

def flow_simulator(
        t,
        Qcom_x,
        dx = 4*1e-5,  # space step (TYP: 4*1e-5) [m];
        steady_factor = 1.0,
        downsampled_rate = 0.001,  # [s]
        fluid = 'fluid_DOW121',
        mixer = 'mixer_ISSM50nozzle',
        pump = 'pump_viscotec_outdated',
        file_name = 'tmp'
):



    if fluid == 'fluid_DOW121':
        from constants import fluid_DOW121 as fld
    else:
        print("Error: fluid not recognized")
        quit()

    if mixer == 'mixer_ISSM50nozzle':
        from constants import mixer_ISSM50nozzle as mix
    elif mixer == 'mixer_ISSM160':
        from constants import mixer_ISSM160 as mix
    elif mixer == 'mixer_pipe160':
        from constants import mixer_pipe160 as mix
    else:
        print("Error: mixer not recognized")
        quit()

    if pump == 'pump_viscotec':
        from constants import pump_viscotec as pmp
    elif pump == 'pump_viscotec_outdated':
        from constants import pump_viscotec_outdated as pmp
    else:
        print("Error: pump not recognized")
        quit()

    # print(mix.KL_SM) # testprint
    # print(pmp.W_GAP) # test


    # dx = 4*1e-5  # space step (TYP: 4*1e-5) [m]
    dt = dx/fld.WAVESPEED  # time step [s]

    x = np.arange(0,mix.L_TOTAL+dx,dx)  # initialize list of space coordinates

    # t, Qcom_x = gen.corner_naive(
    #         a_tool = 100,  # toolhead acceleration [mm/s^2]
    #         v_tool = 25,  # toolhead velocity [mm/s]
    #         dt = dt,  # time step [s]
    #         precorner_dist = 100,  # straight-line distance before the corner [mm] (TYP: 180)
    #         postcorner_dist = 30,  # straight-line distance after the corner [mm]  (TYP: 30)
    #         layer_height = 0.15,  # layer height [mm]
    #         nozzle_diam = 0.25  # diameter of the nozzle [mm]
    #         )

    # steady_factor = 1.0
    Qcom_x = deepcopy(steady_factor*Qcom_x)

    # t, Qcom_x = gen.twostep(
    #         up_flowrate = 0.50,  # step-up inlet flow rate [mL/min]
    #         down_flowrate = 0,  # step-down inlet flow rate [mL/min]
    #         dt = dt,  # time step [s]
    #         tmax = 7,  # maximum simulation time [s]
    #         t_stepup = 1,  # time of step-up command [s] (TYP: 0.0001)
    #         t_stepdown = 4.2,  # time of step-down command [s]
    #         )


    Qo_x = 0*Qcom_x  # initialize outlet flow rate over time horizon

    Po_x = 0*t  # initialize outlet pressure over time horizon

    Pp_x = 0*t  # initialize pump pressure over time horizon

    Pi_t = 0*x  # initialize initial pressure over space horizon

    Qi_t = 0*x  # initialize initial flow rate over space horizon

    Pn_t = deepcopy(Pi_t)  # initialize "new time" pressure over space horizon

    Qn_t = deepcopy(Qi_t)  # initialize "new time" flow rate over space horizon

    Kl_t = 0*x+mix.KL_SM  # length parameter over space horizon is for static mixer

    Kg_t = 0*x+mix.KG  # shear parameter over space horizon for static mixer

    KL_NOZ = mix.D_IN*(mix.D_IN**(3*fld.N_INDEX)-mix.D_OUT**(3*fld.N_INDEX))/(3*fld.N_INDEX*(mix.D_IN-mix.D_OUT)*mix.D_OUT**(3*fld.N_INDEX))  # length parameter of the nozzle [1]
    Kl_t[x>mix.L_SM] = KL_NOZ  # length parameter within the nozzle (last 0.032 m of flow geometry) is nozzle)

    KG_NOZ = 8  # shear parameter for nozzle
    Kg_t[x>mix.L_SM] = KG_NOZ  # shear parameter within the nozzle (last 0.032 m of flow geometry) is nozzle)


    mu = fld.K_INDEX*(Kg_t*np.abs(Qn_t)/(mix.D_MIX*mix.A_MIX)+fld.Z_SHIFT)**(fld.N_INDEX-1)  # effective viscosity (non-newtonian) [Pa*s]

    Fn_t = 32*Kl_t*mu*Qn_t/(fld.RHO*(mix.D_MIX**2)*mix.A_MIX)  # acceleration due to friction over space horizon at initial time step

    w_m = Qcom_x/pmp.EXTRUSION_RATIO  # commanded motor speed over time [rad/s]


    theta_m = 0  # initialize motor displacement (relative to previous time step) [rad]
    theta_r = 0  # initialize rotor displacement (relative to previous time step) [rad]
    d_theta = theta_m-theta_r #initialize (torsional) deflection of elastic coupler [rad]

    w_r = 0*t  # initialize rotor speed over time horizon [rad/s]
    w_r_dot = 0*t  # initialize rotor acceleration over time horizon [rad/s^2]

    # w_r_IC = np.array([0,0,0,0])
    w_r_dot_IC = np.array([0,0,0,0])


    w_r_tol = 1e-9  # velocity tolerance [rad/s]

    V_cav = pmp.N_CAV*0*np.pi*pmp.R_CAV**2

    #%% constants used in solver

    # NN constants
    solve_const = mix.A_MIX/(fld.WAVESPEED*fld.RHO)  #mixer constant

    # Elastic torsion constants
    A_const = 1*(np.pi*pmp.D_C**4*pmp.R*pmp.RHO_C*pmp.T_C)/(16*pmp.MC_C*pmp.L_C*pmp.M_ROTOR*pmp.R_R**2)  # forcing constant
    B_const = 1*(-8*np.pi*pmp.R_SO**2*pmp.L_S)/(pmp.M_ROTOR*(pmp.R_SO**2-pmp.R_R**2))  # viscous friction constant
    C_const = 1*(-2*pmp.N_CAV*pmp.A_CAV*np.sin(np.deg2rad(pmp.PHI)))/(pmp.M_ROTOR*pmp.R_R)  # pressure constant
    D_const = 1*(-6*pmp.RHO_S*pmp.R*pmp.T_S)*(pmp.S*pmp.A_RS*pmp.MU_FRIC)/(pmp.MC_S*pmp.M_ROTOR*pmp.R_R)  # Coloumbic friction constant

    # Backflow constants
    R_gap = (pmp.R_R**2-((pmp.W_GAP**2+pmp.R_R**2-pmp.R_SI**2)/(-2*pmp.W_GAP))**2)**0.5  # half-width (radius) of slip gap [m]
    R_gap = (pmp.R_R**2-((pmp.W_GAP**2+pmp.R_R**2-pmp.R_R**2)/(-2*pmp.W_GAP))**2)**0.5  # half-width (radius) of slip gap [m]
    A_gap = (R_gap*((pmp.R_SI**2-R_gap**2)**0.5-(pmp.R_R**2-R_gap**2)**0.5+2*pmp.W_GAP)
            + pmp.R_SI**2*np.arctan(R_gap/(pmp.R_SI**2-R_gap**2)**0.5)
            - pmp.R_R**2*np.arctan(R_gap/(pmp.R_R**2-R_gap**2)**0.5))  # area of slip gap [m^2]
    U_gap = 2*pmp.R_R*np.arcsin(R_gap/pmp.R_R)+2*pmp.R_SI*np.arcsin(R_gap/pmp.R_SI)  # perimeter of slip gap [m]
    d_h = 4*A_gap/U_gap  # hydraulic diameter [m]
    F_const = A_gap*(((d_h*fld.RHO/(fld.K_INDEX*A_gap))**(1/(fld.N_INDEX+6)))
                *((2*pmp.K_S/pmp.W_GAP)**((1-fld.N_INDEX)/(fld.N_INDEX+6)))
                *((2*d_h*A_gap**2/(0.3164*fld.RHO*pmp.L_GAP))**(4/(fld.N_INDEX+6)))) # backflow constant

    # Elastic expansion constants
    E_const = 3*pmp.RHO_S*pmp.R*pmp.T_S/pmp.MC_S  #  elastic modulus constant
    V_const = np.pi*pmp.L_S*pmp.R_R**2  # rotor volume constant
    P_const = 8*np.pi*pmp.L_S/pmp.A_CAV**2


    #%% functions used in solver

    def torsion(w_r_dot_IC_current,w_r_current,w_m_new,d_theta_current,mu_inlet,Pn_t_inlet):
        w_r_intermediate = w_r_current + dt*(
            55*w_r_dot_IC_current[3]
            - 59*w_r_dot_IC_current[2]
            + 37*w_r_dot_IC_current[1]
            - 9*w_r_dot_IC_current[0])/24 # use finite-difference method to calculate rotor speed from accel [rad/s]
        # theta_m_new = w_m_new*dt-theta_m_current  # use finite-difference method to calculate change in motor displacement [rad]
        # theta_r_new = w_r_new*dt-theta_r_current  # use finite-difference method to calculate change in rotor displacement [rad]
        d_theta_intermediate = (w_m_new-w_r_intermediate)*dt+d_theta_current  # calculate elastic coupler deflection [rad]
        w_r_dot_intermediate = (A_const*d_theta_intermediate
                    + B_const*mu_inlet*w_r_intermediate
                    + C_const*Pn_t_inlet
                    + D_const*np.tanh(w_r_intermediate/w_r_tol))  # rotor acceleration from elastic torsion model [rad/s^2]

        w_r_new = w_r_current + dt*(
            251*w_r_dot_intermediate
            + 646*w_r_dot_IC_current[3]
            - 264*w_r_dot_IC_current[2]
            + 106*w_r_dot_IC_current[1]
            - 19*w_r_dot_IC_current[0]
            )/720
        d_theta_new = (w_m_new-w_r_new)*dt+d_theta_current  # calculate elastic coupler deflection [rad]
        w_r_dot_new = (A_const*d_theta_new
                    + B_const*mu_inlet*w_r_new
                    + C_const*Pn_t_inlet
                    + D_const*np.tanh(w_r_new/w_r_tol))  # rotor acceleration from elastic torsion model [rad/s^2]
        w_r_dot_IC_new = np.roll(w_r_dot_IC_current,-1)
        w_r_dot_IC_new[-1] = w_r_dot_new
        return w_r_new, d_theta_new, w_r_dot_new, w_r_dot_IC_new

    def expansion(w_r_new,Pn_t_inlet,V_cav_current):
        L_cav_new = w_r_new*pmp.EXTRUSION_RATIO*dt/(pmp.N_CAV*pmp.A_CAV)
        d_R_new = pmp.H_S*np.sign(Pn_t_inlet)*min(abs(Pn_t_inlet)/E_const,pmp.S_MAX)  # change in stator wall thickness {-H_S < d_R < H_S} [m]
        d_V_new = pmp.N_CAV*L_cav_new*np.pi*(pmp.R_CAV+d_R_new)**2-V_cav_current  # change in volume of cavities at the inlet [m^3]
        V_cav_new = pmp.N_CAV*L_cav_new*np.pi*(pmp.R_CAV+d_R_new)**2  # volume of cavities at the inlet [m^3]
        return L_cav_new, d_R_new, d_V_new, V_cav_new

    # def backflow(mu_inlet,Qn_t_inlet,Pn_t_inlet):
    #     #TODO: beware of negative delta_P
    #     P_pump = P_const*mu_inlet*Qn_t_inlet
    #     delta_P_new = Pn_t_inlet-P_pump
    #     Q_b_new = np.sign(delta_P_new)*F_const*(abs(delta_P_new)**(4/(fld.N_INDEX+6)))
    #     return delta_P_new, Q_b_new

    def backflow(mu_inlet,Qn_t_inlet,Pn_t_inlet):
        P_pump = P_const*mu_inlet*Qn_t_inlet
        delta_P_new = np.max([Pn_t_inlet-P_pump,0])
        Q_b_new = F_const*(abs(delta_P_new)**(4/(fld.N_INDEX+6)))
        return delta_P_new, Q_b_new

    def copy_PQF(Pn_t_current,Qn_t_current,Fn_t_current):
        Pc_t_new = deepcopy(Pn_t_current)  # "current time" pressure equals previous "new time" pressure
        Qc_t_new = deepcopy(Qn_t_current)  # "current time" flow rate equals previous "new time" flow rate
        Fc_t_new = deepcopy(Fn_t_current)  # "current time" accel due to friction equals previous "new time" friction term
        return Pc_t_new, Qc_t_new, Fc_t_new

    def inlet_PQ(w_r_new,Q_b_new,d_V_new,Qc_t_postinlet,Pc_t_postinlet,Fc_t_postinlet):
        Qn_t_inlet = w_r_new*pmp.EXTRUSION_RATIO-Q_b_new-d_V_new/dt  # inlet flow rate at "new time" calculated from current rotor velocity
        Pn_t_inlet = (Qn_t_inlet-Qc_t_postinlet+solve_const*Pc_t_postinlet+mix.A_MIX*Fc_t_postinlet*dt)/solve_const  # inlet pressure at "new time" from CM
        return Qn_t_inlet, Pn_t_inlet

    def pipe_PQ(Pn_t_new,Qn_t_new,Qc_t_new,Pc_t_new,Fc_t_new):
        # entire space horizon computed for current time step:
        Pn_t_new[1:-1] = (Qc_t_new[:-2]-Qc_t_new[2:]+solve_const*Pc_t_new[:-2]+solve_const*Pc_t_new[2:]-mix.A_MIX*Fc_t_new[:-2]*dt+mix.A_MIX*Fc_t_new[2:]*dt)/(2*solve_const)
        Qn_t_new[1:-1] = Qc_t_new[:-2]+solve_const*Pc_t_new[:-2]-mix.A_MIX*Fc_t_new[:-2]*dt-solve_const*Pn_t_new[1:-1]
        return Pn_t_new, Qn_t_new

    def outlet_PQ(Po_x_new,Qc_t_preoutlet,Pc_t_preoutlet,Fc_t_preoutlet):
        Pn_t_outlet = Po_x_new  # pressure at "new time" at the outlet is atmospheric pressure (TYP: 0)
        Qn_t_outlet = Qc_t_preoutlet+solve_const*Pc_t_preoutlet-mix.A_MIX*Fc_t_preoutlet*dt-solve_const*Pn_t_outlet  # outlet flowrate at "new time" via CM
        Qo_x_new = Qn_t_outlet   # outlet flowrate at "new time" via CM
        return Pn_t_outlet, Qn_t_outlet, Qo_x_new

    def pipe_F(Qn_t_new):
        # Q_smooth = (Qn_t_new[:-1:2]+Qn_t_new[1::2])/2  # smoothness enforcement
        # Qn_t_new[:-1:2] = Q_smooth
        # Qn_t_new[1::2] = Q_smooth
        # Qn_t_new[-1] = Q_smooth[-1]
        mu_new = fld.K_INDEX*(Kg_t*np.abs(Qn_t_new)/(mix.D_MIX*mix.A_MIX)+fld.Z_SHIFT)**(fld.N_INDEX-1)  # non-newtonian viscosity power law
        Fn_t_new = 32*Kl_t*mu_new*Qn_t_new/(fld.RHO*(mix.D_MIX**2)*mix.A_MIX)  # acceleration due to newtonian friction over space horizon at new time step
        return mu_new, Fn_t_new


    #%%

    def progress_update():
        if (j/update_loops)==round(j/update_loops):  # progress update
            print(str(round(100*j/len(t),2))
                  + '% complete, '
                  + str(round(((perf_counter()-runtime_current)*(len(t)-j)/update_loops)/60))
                  + ' minutes remaining')  # progress update printout
            runtime_current_new = perf_counter()  # update runtime counter
            #todo: this is borken but wtv
            return runtime_current_new

        print("If you can see this message, then this function is running correctly.")

    def test_prints():
        # Q = 0.25*1.6666667e-08
        # v = Q/mix.A_MIX
        # mu = fld.K_INDEX*(mix.KG*np.abs(Q)/(mix.D_MIX*mix.A_MIX)+fld.Z_SHIFT)**(fld.N_INDEX-1)
        # f_d = mix.KL_SM*64*mu*mix.A_MIX/(fld.RHO*Q*mix.D_OUT)
        # delta_p = f_d*mix.L_SM*fld.RHO*v**2/(2*mix.D_OUT)
        # delta_p*0.000145038

        print('\n\nFlowrate Results:\n')
        print('Q           = +' + str(round(Qo_x[-1] / 1.6666667e-08, 4)) + ' mL/min\n')
        print('Q_torsion   = +' + str(round(w_r[-1] * pmp.EXTRUSION_RATIO / 1.6666667e-08, 4)) + ' mL/min')
        print('Q_backflow  = ' + str(round(-1 * Q_b / 1.6666667e-08, 4)) + ' mL/min')
        print('Q_expansion = ' + str(round(-1 * (d_V / dt) / 1.6666667e-08, 4)) + ' mL/min')
        print('\n\nRotor Results:\n')
        print('w   = ' + str("{:e}".format(round(w_r_dot[-1]))) + ' rad/s^2\n')
        print('w_A = +' + str("{:e}".format(round(A_const * d_theta))) + ' forcing')
        print('w_B = ' + str("{:e}".format(round(B_const * mu[200] * w_r[-1]))) + ' viscous')
        print('w_C = ' + str("{:e}".format(round(C_const * Pn_t[0]))) + ' pressure')
        print('w_D = ' + str("{:e}".format(round(D_const * np.tanh(w_r[-1] / w_r_tol)))) + ' coloumbic\n')

        # print('check Q_com_x factor')

    def test_plot():
        print('yep')
        # import matplotlib
        # class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        #     def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        #         self.oom = order
        #         self.fformat = fformat
        #         matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
        #     def _set_order_of_magnitude(self):
        #         self.orderOfMagnitude = self.oom
        #     def _set_format(self, vmin=None, vmax=None):
        #         self.format = self.fformat
        #         if self._useMathText:
        #             self.format = r'$\mathdefault{%s}$' % self.format

        font = {'family': 'serif',
                'color': 'k',
                'weight': 'normal',
                'size': 14,
                }

        # marker = itertools.cycle((',', '.', 'o', '*'))
        # color = itertools.cycle(('b','r','g','k'))

        # %% figure: outlet flowrate
        # plot time horizon of outlet flow rate and commanded (inlet) flow rate

        fig_flow = plt.figure("flowrate graph")
        ax = fig_flow.add_subplot(1, 1, 1)

        plt.xlabel('Time, t [s]', fontdict=font)
        plt.ylabel('Output Flowrate, Q_o [psi]', fontdict=font)

        # plt.xlim(-1,31)
        # plt.ylim(0,500)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(t[::round(downsampled_rate / dt)], Qcom_x[::round(downsampled_rate / dt)] / 1.6666667e-08,
                 color='r', linewidth=2, linestyle='--',
                 label='Flow Rate Command, Q_c')  # plot inlet (commanded) flow
        plt.plot(t[::round(downsampled_rate / dt)], Qo_x[::round(downsampled_rate / dt)] / 1.6666667e-08,
                 color='b', linewidth=2,
                 label='Flow Rate Output, Q_o [mL/min]')  # plot outlet flow

        # plt.plot(rise_fit[0],rise_fit[1]/1.6666667e-08,
        #           color = 'k',linewidth = 2, linestyle = '--',
        #           label = 'gomp fit rise') #plot outlet flow

        # plt.plot(fall_fit[0],fall_fit[1]/1.6666667e-08,
        #           color = 'k',linewidth = 2, linestyle = '--',
        #           label = 'gomp fit fall') #plot outlet flow

        # plt.plot(t[::round(downsampled_rate/dt)],Pp_x[::round(downsampled_rate/dt)]*0.000145038,
        #           color = 'k',linewidth = 2,linestyle = '-',
        #           label = 'Predicted Pump Pressure, P_p') #plot inlet (commanded) flow

        # test1 = deepcopy(Qo_x[:-1:2])
        # test2 = deepcopy(Qo_x[1::2])

        # Qo_smooth = (Qo_x[:-1:2]+Qo_x[1::2])/2
        # t_smooth = t[1::2]

        # plt.plot(t,(Qcom_x/1)/1.6666667e-08,
        #           color = 'r',linewidth = 2,linestyle = '--',
        #           label = 'Flow Rate Command, Q_c') #plot inlet (commanded) flow
        # plt.plot(t,Qo_x/1.6666667e-08,
        #           color = 'b',linewidth = 2,
        #           label = 'Flow Rate Output, Q_o [mL/min]') #plot outlet flow

        # plt.plot(t_smooth,Qo_smooth/1.6666667e-08,
        #           color = 'g',linewidth = 2,
        #           label = 'Flow Rate Output, Q_o [mL/min]') #plot outlet flow

        # plt.plot(t,Qcom_x/1.6666667e-08,
        #           color = 'r',linewidth = 2,linestyle = '--',
        #           label = 'Flow Rate Command, Q_c') #plot inlet (commanded) flow

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_flow = plt.figure("flowrate legend")
        # leg_flow.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])

        # %% save figs

        # fig_flow.savefig('flowrate_CM_NN_torsion.png',bbox_inches='tight',dpi=600)
        # leg_flow.savefig('flowrate_CM_NN_torsion_legend.png',bbox_inches='tight',dpi=600)

    def save_workspace_vars(workspace_vars, base_path = '/Users/james/Desktop/sim_samples/'):
        plt.close('all')
        # workspace_vars = 'CM_combined_workspace_vars'
        # workspace_vars = 'mixer_0.75_mlmin_7s'

        np.savez(base_path + workspace_vars + '.npz',
                 Qo_x=Qo_x[::round(downsampled_rate / dt)],
                 Qcom_x=Qcom_x[::round(downsampled_rate / dt)],
                 Pp_x=Pp_x[::round(downsampled_rate / dt)],
                 time=t[::round(downsampled_rate / dt)])



    #%%

    update_loops = 5000000  # number of loops per progress update
    print("Number of Loops: "+str(len(t)))
    runtime_current = perf_counter()


    # P = np.zeros([len(Pn_t),len(Po_x)])
    # Q = np.zeros([len(Qn_t),len(Po_x)])
    # F = np.zeros([len(Fn_t),len(Po_x)])


    for j in range(1,len(t)):  # for each time step, advance space horizons by one time step


        w_r[j], d_theta, w_r_dot[j], w_r_dot_IC = torsion(w_r_dot_IC,w_r[j-1],w_m[j],d_theta,mu[200],Pn_t[0])
        L_cav, d_R, d_V, V_cav = expansion(w_r[j],Pn_t[0],V_cav)
        delta_P, Q_b = backflow(mu[200],Qn_t[0],Pn_t[0])
        Pc_t, Qc_t, Fc_t = copy_PQF(Pn_t,Qn_t,Fn_t)

        Pn_t,Qn_t = pipe_PQ(Pn_t,Qn_t,Qc_t,Pc_t,Fc_t)
        Qn_t[0], Pn_t[0] = inlet_PQ(w_r[j],Q_b,d_V,Qc_t[1],Pc_t[1],Fc_t[1])
        Pn_t[-1], Qn_t[-1], Qo_x[j] = outlet_PQ(Po_x[j],Qc_t[-2],Pc_t[-2],Fc_t[-2])

        mu, Fn_t = pipe_F(Qn_t)

        Pp_x[j] = Pn_t[0]

        if (j/update_loops)==round(j/update_loops):  # progress update
            print(str(round(100*j/len(t),2))
                  + '% complete, '
                  + str(round(((perf_counter()-runtime_current)*(len(t)-j)/update_loops)/60))
                  + ' minutes remaining')  # progress update printout
            runtime_current = perf_counter()  # update runtime counter



        # P = [P;Pn_t]
        # Q = [Q;Qn_t]
        # F = [F;Fn_t]


    print('Solving completed')
    # test_prints()
    test_plot()

    save_workspace_vars(file_name)


    return (t[::round(downsampled_rate / dt)],
            Qcom_x[::round(downsampled_rate / dt)],
            Qo_x[::round(downsampled_rate / dt)],
            Pp_x[::round(downsampled_rate / dt)])

#%% rise and fall time calculation

def rise_gomp(t,Q_outlet):

    # window_size = int(np.floor((0.01*(1/np.diff(t).mean()))/2)*2+1)  # window size equals 0.01 seconds, expressed as a number of time steps, rounded to the nearest odd natural
    # data_smooth = medfilt(sg_filter(Q_outlet,window_size,3),kernel_size=window_size)
    stop_idx = len(Q_outlet) - np.argmax(Q_outlet[::-1]) - 1
    # stop_idx = np.argmax(data_smooth)+np.argmin(np.abs(data_smooth[np.argmax(data_smooth):]-(np.max(data_smooth)-sensor_data.iloc[np.argmax(data_smooth),2])))

    # rise_fit = sensor_data[:stop_idx]


    x = t[:stop_idx]
    y = Q_outlet[:stop_idx]


    a = np.max(Q_outlet)

    def objective(x,b,c):
        return a*np.exp(-1*b*np.exp(-1*c*x))

    # param, param_cov = curve_fit(objective,x,y,p0=[4.5e6,10,10],maxfev = 100000)
    param, param_cov = curve_fit(objective,x,y,p0=[1,10],maxfev = 100000)
    b,c = param
    rise_fit = np.array([x,objective(x,b,c)])

    rise_05percent = np.log(np.log(0.05)/(-1*b))/(-1*c)
    rise_95percent = np.log(np.log(0.95)/(-1*b))/(-1*c)

    return rise_fit, rise_05percent, rise_95percent, a, b, c

def fall_gomp(t,Q_outlet):

    # window_size = int(np.floor((0.01*(1/sensor_data['time'].diff().mean()))/2)*2+1)  # window size equals 0.01 seconds, expressed as a number of time steps, rounded to the nearest odd natural
    # data_smooth = medfilt(sg_filter(sensor_data.iloc[:,1],window_size,3),kernel_size=window_size)
    start_idx = np.argmax(Q_outlet)
    # start_idx = np.argmin(np.abs(data_smooth[:np.argmax(data_smooth)+1]-(np.max(data_smooth)-sensor_data.iloc[np.argmax(data_smooth),2])))

    # fall_fit = sensor_data[start_idx:]

    x = t[start_idx:]
    y = Q_outlet[start_idx:]

    a = np.max(Q_outlet)

    def objective(x,b,c):
        return a*(1-np.exp(-1*b*np.exp(-1*c*x)))

    # param, param_cov = curve_fit(objective,x,y,p0=[4.5e6,10,1],maxfev = 100000)
    param, param_cov = curve_fit(objective,x,y,p0=[100,1],maxfev = 100000)
    b,c = param
    fall_fit = np.array([x,objective(x,b,c)])

    fall_05percent = np.log(np.log(1-0.05)/(-1*b))/(-1*c)
    fall_95percent = np.log(np.log(1-0.95)/(-1*b))/(-1*c)

    return fall_fit, fall_05percent, fall_95percent, a, b, c

def rise_timer(t,Q_outlet):

    curve_percentage = 0.05

    Q_outlet[np.argmax(Q_outlet)+1:] = 0
    rise_t1 = t[np.argmin(np.abs(Q_outlet-curve_percentage*np.max(Q_outlet)))]
    rise_t2 = t[np.argmin(np.abs(Q_outlet-(1-curve_percentage)*np.max(Q_outlet)))]
    rise_time = rise_t2 - rise_t1

    return rise_t1, rise_t2, rise_time

def fall_timer(t,Q_outlet):

    curve_percentage = 0.05

    Q_outlet[:np.argmax(Q_outlet)] = 0
    fall_t1 = t[np.argmin(np.abs(Q_outlet-(1-curve_percentage)*np.max(Q_outlet)))]
    fall_t2 = t[np.argmin(np.abs(Q_outlet-curve_percentage*np.max(Q_outlet)))]
    fall_time = fall_t2 - fall_t1

    return fall_t1, fall_t2, fall_time

# rise_t1, rise_t2, rise_time = rise_timer(t,deepcopy(Qo_x))
# fall_t1, fall_t2, fall_time = fall_timer(t,deepcopy(Qo_x))

# downsampled_rate = 0.001  # [s]
# rise_fit, rise_05percent_gomp, rise_95percent_gomp, a_rise, b_rise, c_rise = rise_gomp(t[::round(downsampled_rate/dt)],deepcopy(Qo_x[::round(downsampled_rate/dt)]))
# fall_fit, fall_05percent_gomp, fall_95percent_gomp, a_fall, b_fall, c_fall = fall_gomp(t[::round(downsampled_rate/dt)],deepcopy(Qo_x[::round(downsampled_rate/dt)]))

    
#%% save workspace variables




    # saved_data = np.load(workspace_vars+'.npz')
    # Qo_x, Qcom_x, Pp_x, t = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time']




#%% testing

# do if name == main
if __name__ == '__main__':
    dx = 4 * 1e-5  # space step (TYP: 4*1e-5) [m]
    dt = dx / fld.WAVESPEED  # time step [s]
    t, Qcom_x, _ = gen.corner_naive(
        a_tool=100,  # toolhead acceleration [mm/s^2]
        v_tool=25,  # toolhead velocity [mm/s]
        dt=dt,  # time step [s]
        precorner_dist=100,  # straight-line distance before the corner [mm] (TYP: 180)
        postcorner_dist=30,  # straight-line distance after the corner [mm]  (TYP: 30)
        layer_height=0.15,  # layer height [mm]
        nozzle_diam=0.25  # diameter of the nozzle [mm]
    )
    t, Qcom_x, Qo_x, Pp_x = flow_simulator(t, Qcom_x)
