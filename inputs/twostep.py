# import re
# import itertools
import numpy as np
# import jax.numpy as jnp
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

def twostep(
        dt=0.001,  # time step [s]
        tmax=50,  # maximum simulation time [s]
        flowrate_up = 1,  # step-up inlet flow rate [mL/min]
        flowrate_down = 0.001,  # step-down inlet flow rate [mL/min]
        t_flowup = 0.0001,  # time of step-up command [s]
        t_flowdown = 7.5,  # time of step-down command [s]
        beadwidth_down=0.0001,  # step-down bead width [m]
        beadwidth_up = 0.0029, # step-up bead width [m]
        t_beaddown = 10,  # time of step-up command [s]
        t_beadup = 11,  # time of step-down command [s]
        ):

    ts = np.arange(0,tmax,dt)  # initialize list of time coordinates

    input_Q = np.ones(np.shape(ts))*flowrate_down/6e7 # inlet flow rate: 0 mL/min at t=0
    input_Q[ts>t_flowup] = flowrate_up/6e7  # inlet flow rate: 1 mL/min at t=0.0001 (step up), converted to [m^3/s]
    input_Q[ts>t_flowdown] = flowrate_down/6e7  # inlet flow rate: 0 mL/min at t=10 (step down), converted to [m^3/s]

    input_W = beadwidth_up*np.ones(np.shape(ts))
    input_W[ts > t_beaddown] = beadwidth_down
    input_W[ts > t_beadup] = beadwidth_up

    print('Simulated duration: ' + str(round(tmax,2)) + ' seconds.')     

    # ts_jax = jnp.asarray(ts)
    # input_Q_jax = jnp.asarray(input_Q)
    # input_W_jax = jnp.asarray(input_W)

    return ts, input_Q, input_W
    # return ts_jax, input_Q_jax, input_W_jax

#%% 

 
#%% testing

if __name__ == "__main__": 

    t,Q_com = twostep(1)
    
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
    plt.ylabel('Commanded Flowrate, $Q_{com}$ [m^3/s]',fontdict = font)
    
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

