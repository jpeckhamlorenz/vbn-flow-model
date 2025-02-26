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
# from scipy.optimize import curve_fit
# import cv2 as cv
# import torch
# import torchvision
# from segment_anything import sam_model_registry, SamPredictor
# import gc
# from time import perf_counter
# import os.path
# from copy import deepcopy

#%%

def corner_naive(
        dt = 0.01,  # time step [s]
        a_tool = 100,  # toolhead acceleration [mm/s^2]
        v_tool = 25,  # toolhead velocity [mm/s]
        precorner_dist = 180,  # straight-line distance before the corner [mm]
        postcorner_dist = 20,  # straight-line distance after the corner [mm]
        layer_height = 0.15,  # layer height [mm]
        nozzle_diam = 0.25  # diameter of the nozzle [mm]
        ):

    
    acceleration_dist = (v_tool**2)/(2*a_tool)  # distance traveled while decelerating/accelerating between v_tool and 0mm/s [mm]
    
    precorner_steady_duration = (precorner_dist-acceleration_dist)/v_tool  # duration of steady toolhead velocity [s]
    precorner_decel_duration = v_tool/a_tool  # duration of decreasing toolhead velocity [s]
    postcorner_accel_duration = v_tool/a_tool  # duration of increasing toolhead velocity [s]
    postcorner_steady_duration = (postcorner_dist-acceleration_dist)/v_tool  # duration of steady toolhead velocity [s]
    
    duration = (precorner_steady_duration
                + precorner_decel_duration
                + postcorner_steady_duration
                + postcorner_accel_duration)  # [s]
    
    
    #%%
    
    t = np.arange(0,duration+dt,dt)  # initialize list of time coordinates [s]
    
    V_com = (
        v_tool
        )*np.ones(np.shape(t))  # commanded toolhead velocity: steady pre-corner [mm/s]
    V_com[np.logical_and(
          t>=precorner_steady_duration, 
          t<=precorner_steady_duration+precorner_decel_duration
          )] = np.linspace(v_tool,0,np.sum(np.logical_and(
                t>=precorner_steady_duration, 
                t<=precorner_steady_duration+precorner_decel_duration
                )))  # commanded toolhead velocity: transient pre-corner (decelerating) [mm/s]
    V_com[np.logical_and(
          t>=precorner_steady_duration+precorner_decel_duration, 
          t<=precorner_steady_duration+precorner_decel_duration+postcorner_accel_duration
          )] = np.linspace(0,v_tool,np.sum(np.logical_and(
                t>=precorner_steady_duration+precorner_decel_duration, 
                t<=precorner_steady_duration+precorner_decel_duration+postcorner_accel_duration
                )))  # commanded toolhead velocity: transient post-corner (accelerating) [mm/s]
    
    Q_com = V_com*layer_height*nozzle_diam*1e-09     
    Q_com[t<precorner_steady_duration/16] =  1*1.6666667e-08  # flow rate: 1 mL/min, converted to [m^3/s]
    Q_com[t<0.001] = 0*1e-09

    input_W = nozzle_diam * np.ones(np.shape(t)) / 1000  # diameters of the nozzle [m]
    input_W[t>precorner_steady_duration/16] = np.max(input_W[t>precorner_steady_duration/16])*Q_com[t>precorner_steady_duration/16] / np.max(Q_com[t>precorner_steady_duration/16])  # diameters of the nozzle [m]


    print('Simulated duration: ' + str(round(duration,2)) + ' seconds.')     


    
    
    
    return t, Q_com, input_W
    # return AB_dist, AB_time, ApB_dist, ApB_time, BCp_dist, BCp_time, BC_dist, BC_time


 
#%% testing

if __name__ == "__main__": 

    t, Q_com, input_beadwidth = corner_naive(
        dt = 0.01,  # time step [s]
        a_tool = 100,  # toolhead acceleration [mm/s^2]
        v_tool = 25,  # toolhead velocity [mm/s]
        precorner_dist = 180,  # straight-line distance before the corner [mm]
        postcorner_dist = 20,  # straight-line distance after the corner [mm]
        layer_height = 0.15,  # layer height [mm]
        nozzle_diam = 0.25  # diameter of the nozzle [mm]
        )
    
    # AB_dist, AB_time, ApB_dist, ApB_time, BCp_dist, BCp_time, BC_dist, BC_time = corner_naive(
    #         a_tool = 250,  # toolhead acceleration [mm/s^2]
    #         v_tool = 25,  # toolhead velocity [mm/s]
    #         dt = 4*1e-5/880,  # time step [s]
    #         precorner_dist = 100,  # straight-line distance before the corner [mm] (TYP: 180)
    #         postcorner_dist = 25,  # straight-line distance after the corner [mm]  (TYP: 30)
    #         layer_height = 0.15,  # layer height [mm]
    #         nozzle_diam = 0.25  # diameter of the nozzle [mm]
    #         )
    #%% plotting setup 
    
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
            'color':  'k',
            'weight': 'normal',
            'size': 14,
            }
    
    # marker = itertools.cycle((',', '.', 'o', '*')) 
    # color = itertools.cycle(('b','r','g','k'))
    
    
    #%% figure: test
    #plot of 
        
    fig_test = plt.figure("test graph")
    ax = fig_test.add_subplot(1,1,1)
    
    
    plt.xlabel('Time, $t$ [s]',fontdict = font)
    plt.ylabel('Commanded Flowrate, $Q_{com}$ [m$^3$/s]',fontdict = font)
    
    # plt.xlim(0,1)
    # plt.ylim(0,30)
    
    plt.grid(which='major',visible=True,color='0.5',linestyle='-',linewidth=0.5)
    
    plt.xscale('linear')
    plt.yscale('linear')
    
    
    
    plt.plot(t,Q_com,
             color = 'k',linewidth = 2,linestyle = '-',
             label = 'test') #plot inlet (commanded) flow   
    
    # ax.set_xticks(np.arange(0,1.1,0.1))
    # ax.set_yticks(np.arange(0,30,5))
    
    # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)
    
    # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)
    
    # ax.legend()
    
    # leg_test = plt.figure("test legend")
    # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])

    plt.figure()
    plt.plot(t,input_beadwidth)

#%%
