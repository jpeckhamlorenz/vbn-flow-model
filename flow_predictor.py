# flow prediction function for VBN
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import torch




#%%

def flow_predictor(
            ts,
            input_flowrate,
            input_beadwidth,
            IC = np.array([0.0, 0.0]),
            fluid  = 'fluid_DOW121',
            mixer  = 'mixer_ISSM50nozzle',
            pump   = 'pump_viscotec_outdated'
    ):

    # from constants import fluid, mixer, and pump parameters

    if fluid == 'fluid_DOW121':
        from constants import fluid_DOW121 as fld
    else:
        raise SystemExit("Error: fluid not recognized")

    if mixer == 'mixer_ISSM50nozzle':
        from constants import mixer_ISSM50nozzle as mix
    elif mixer == 'mixer_ISSM160':
        from constants import mixer_ISSM160 as mix
    elif mixer == 'mixer_pipe160':
        from constants import mixer_pipe160 as mix
    else:
        raise SystemExit("Error: mixer not recognized")

    if pump == 'pump_viscotec':
        from constants import pump_viscotec as pmp
    elif pump == 'pump_viscotec_outdated':
        from constants import pump_viscotec_outdated as pmp
    else:
        raise SystemExit("Error: pump not recognized")

    # %%
    input_motor = integrate.cumulative_trapezoid(input_flowrate / pmp.EXTRUSION_RATIO, ts, initial=0.0)

    # %% constants used in solver

    # Elastic torsion constants
    A_const = 1 * (np.pi * pmp.D_C ** 4 * pmp.R * pmp.RHO_C * pmp.T_C) / (
                16 * pmp.MC_C * pmp.L_C * pmp.M_ROTOR * pmp.R_R ** 2)  # forcing constant
    B_const = 1 * (-8 * np.pi * pmp.R_SO ** 2 * pmp.L_S) / (
                pmp.M_ROTOR * (pmp.R_SO ** 2 - pmp.R_R ** 2))  # viscous friction constant
    C_const = 1 * (-2 * pmp.N_CAV * pmp.A_CAV * np.sin(np.deg2rad(pmp.PHI))) / (
                pmp.M_ROTOR * pmp.R_R)  # pressure constant
    D_const = 1 * (-6 * pmp.RHO_S * pmp.R * pmp.T_S) * (pmp.S * pmp.A_RS * pmp.MU_FRIC) / (
                pmp.MC_S * pmp.M_ROTOR * pmp.R_R)  # Coloumbic friction constant

    # Mixer pressure constants
    N_const = (128 * fld.K_INDEX * mix.L_NOZ * pmp.EXTRUSION_RATIO ** fld.N_INDEX) / (
                3 * fld.N_INDEX * (3 * np.pi) ** fld.N_INDEX * mix.KG ** (1 - fld.N_INDEX))
    M_const = (128 * fld.K_INDEX * mix.KL_SM * pmp.EXTRUSION_RATIO ** fld.N_INDEX * np.pi ** (1 - fld.N_INDEX)) / (
                np.pi * mix.D_IN ** (3 * fld.N_INDEX + 1) * (4 * mix.KG) ** (1 - fld.N_INDEX))
    U_const = fld.K_INDEX * (np.pi * mix.D_MIX ** 3 / (4 * mix.KG * pmp.EXTRUSION_RATIO)) ** (1 - fld.N_INDEX)

    # Sigmoidal constants
    a = 1 / (pmp.EXTRUSION_RATIO * 6e7)
    b = 1000
    c = 0.1
    d = 0.01

    constants = np.array(
        [A_const, B_const, C_const, D_const, N_const, M_const, U_const, fld.N_INDEX, mix.D_IN, a, b, c, d])

    # %%

    eng = matlab.engine.start_matlab()
    [t, x] = eng.VBN_flow_model_solver(ts, input_motor, input_beadwidth, IC, constants, nargout=2)
    eng.quit()

    t = np.array(t).ravel()
    # t = np.array(t).flatten()
    x = np.array(x)

    W_com = np.interp(t, ts, input_beadwidth)
    Q_com = np.interp(t, ts, input_flowrate)
    Q_out = np.array(x[:, 1:] * pmp.EXTRUSION_RATIO).ravel()


    return t, W_com, Q_com, Q_out


def flow_predictor_plots(
        ts,
        Q_out,
        Q_com,
        W_com
):
    # %% plotting setup

    # plt.close('all')

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
    plt.ylabel('Output Flowrate [mL/min]', fontdict=font)

    # plt.xlim(-1,31)
    # plt.ylim(0,0.08)

    plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

    plt.xscale('linear')
    plt.yscale('linear')

    plt.plot(ts, Q_out * 6e7,
             color='b', linewidth=2,
             label='Flow Rate Output [mL/min]')  # plot outlet flow

    plt.plot(ts, Q_com * 6e7,
             color='r', linewidth=2, linestyle='--',
             label='Flow Rate Command')  # plot inlet (commanded) flow

    # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

    # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

    # ax.legend()

    leg_flow = plt.figure("flowrate legend")
    leg_flow.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

    # %% figure: outlet bead width
    # plot time horizon of outlet bead width and commanded bead width

    fig_bead = plt.figure("beadwidth graph")
    ax = fig_bead.add_subplot(1, 1, 1)

    plt.xlabel('Time, t [s]', fontdict=font)
    plt.ylabel('Bead Width [mm]', fontdict=font)

    # plt.xlim(-1,31)
    plt.ylim(0, 3)

    plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

    plt.xscale('linear')
    plt.yscale('linear')

    # plt.plot(t, x[:,0],
    #             color='b', linewidth=2,
    #             label='Bead Width Output, W_o [m]')  # plot outlet bead width

    plt.plot(ts, W_com * 1000,
             color='k', linewidth=2, linestyle='--',
             label='Bead Width Command')  # plot inlet (commanded) bead width

    # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

    # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

    # ax.legend()

    # leg_bead = plt.figure("beadwidth legend")
    # leg_bead.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])



class ResidualModel(torch.nn.Module):
    def __init__(self):
        super(ResidualModel, self).__init__()
        # self.fc1 = torch.nn.Linear(2, 64)  # Input layer
        # self.fc2 = torch.nn.Linear(64, 256)  # Hidden layer
        # self.fc3 = torch.nn.Linear(256, 64)  # Output layer
        # self.fc4 = torch.nn.Linear(64, 1)  # Output layer

        self.fc1 = torch.nn.Linear(2, 64)  # Input layer
        self.fc2 = torch.nn.Linear(64, 512)  # Hidden layer
        self.fc3 = torch.nn.Linear(512, 1024)  # Hidden layer
        self.fc4 = torch.nn.Linear(1024, 512)  # Output layer
        self.fc5 = torch.nn.Linear(512, 64)  # Output layer
        self.fc6 = torch.nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)

    def traj_correction(self, traj_input, Q_out_prior):
        self.load_state_dict(torch.load('trajectory_correction_DNN_state_v1.pth', weights_only = True))
        self.eval()
        with torch.no_grad():
            output = self(traj_input) * 1e-9
        Q_out_correction = (output).cpu().numpy().reshape(-1)
        Q_out_total = Q_out_prior + Q_out_correction

        return Q_out_total, Q_out_correction

