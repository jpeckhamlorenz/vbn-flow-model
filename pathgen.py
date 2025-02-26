import numpy as np
import matplotlib.pyplot as plt
from flow_predictor import flow_predictor, flow_predictor_plots
from copy import deepcopy
import matplotlib


#%%

# %% plotting setup

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


font = {'family': 'serif',
        'color': 'k',
        'weight': 'normal',
        'size': 14,
        }

# marker = itertools.cycle((',', '.', 'o', '*'))
# color = itertools.cycle(('b','r','g','k'))


#%%
class corner_naive:
    def __init__(self,
                 dt = 0.01,  # time step [s]
                 a_tool = 100,  # toolhead acceleration [mm/s^2]
                 v_tool = 25,  # toolhead velocity [mm/s]
                 precorner_dist = 180,  # straight-line distance before the corner [mm]
                 postcorner_dist = 20,  # straight-line distance after the corner [mm]
                 layer_height = 0.15,  # layer height [mm]
                 nominal_beadwidth = 0.25,  # diameter of the nozzle [mm],
                 steady_factor = 1.00,  # factor by which to multiply the steady-state flow rate (TYP: 1.00)
                 fluid = 'fluid_DOW121',
                 mixer = 'mixer_ISSM50nozzle',
                 pump = 'pump_viscotec_outdated'
                 ):
        # Parameters
        ##############################
        ##############################
        self.dt = dt  # time step [s]
        self.a_tool = a_tool  # toolhead acceleration [mm/s^2]
        self.v_tool = v_tool  # toolhead velocity [mm/s]
        self.precorner_dist = precorner_dist  # straight-line distance before the corner [mm]
        self.postcorner_dist = postcorner_dist  # straight-line distance after the corner [mm]
        self.layer_height = layer_height  # layer height [mm]
        self.nominal_beadwidth = nominal_beadwidth  # diameter of the nozzle [mm]
        self.steady_factor = steady_factor # factor by which to multiply the steady-state flow rate (TYP: 1.00)
        self.fluid = fluid
        self.mixer = mixer
        self.pump = pump




    def pathgen(self):

        acceleration_dist = (self.v_tool ** 2) / (2 * self.a_tool)  # distance traveled while decelerating/accelerating between v_tool and 0mm/s [mm]

        self.precorner_steady_duration = (self.precorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]
        self.precorner_decel_duration = self.v_tool / self.a_tool  # duration of decreasing toolhead velocity [s]
        self.postcorner_accel_duration = self.v_tool / self.a_tool  # duration of increasing toolhead velocity [s]
        self.postcorner_steady_duration = (self.postcorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]

        self.duration = (self.precorner_steady_duration
                    + self.precorner_decel_duration
                    + self.postcorner_steady_duration
                    + self.postcorner_accel_duration)  # [s]

        # %%

        t = np.arange(0, self.duration + self.dt, self.dt)  # initialize list of time coordinates [s]

        V_com = (self.v_tool) * np.ones(np.shape(t))  # commanded toolhead velocity: steady pre-corner [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration
            )] = np.linspace(self.v_tool, 0, np.sum(np.logical_and(
                    t >= self.precorner_steady_duration,
                    t <= self.precorner_steady_duration + self.precorner_decel_duration
                    )))  # commanded toolhead velocity: transient pre-corner (decelerating) [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration + self.precorner_decel_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
            )] = np.linspace(0, self.v_tool, np.sum(np.logical_and(
                t >= self.precorner_steady_duration + self.precorner_decel_duration,
                t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
                )))  # commanded toolhead velocity: transient post-corner (accelerating) [mm/s]

        Q_com = V_com * self.layer_height * self.nominal_beadwidth * 1e-09
        Q_com[t < self.precorner_steady_duration / 16] = 1 * 1.6666667e-08  # flow rate: 1 mL/min, converted to [m^3/s]
        Q_com[t < 0.001] = 0 * 1e-09

        input_W = self.nominal_beadwidth * np.ones(np.shape(t)) / 1000  # diameters of the nozzle [m]
        # input_W[t > precorner_steady_duration / 16] = np.max(input_W[t > precorner_steady_duration / 16]) * Q_com[
        #     t > precorner_steady_duration / 16] / np.max(
        #     Q_com[t > precorner_steady_duration / 16])  # diameters of the nozzle [m]

        print('Simulated duration: ' + str(round(self.duration, 2)) + ' seconds.')

        self.input_t = t
        self.input_Q = Q_com*self.steady_factor
        self.input_W = input_W

        return self.input_t, self.input_Q, self.input_W

    def flow_predictor(self):
        ts, W_com, Q_com, Q_out = flow_predictor(
            ts=self.input_t,
            input_flowrate=self.input_Q,
            input_beadwidth=self.input_W,
            fluid = self.fluid,
            mixer = self.mixer,
            pump = self.pump
        )

        self.ts = ts
        self.W_com = W_com
        self.Q_com = Q_com
        self.Q_out = Q_out

        return self.ts, self.W_com, self.Q_com, self.Q_out

    def load_sim_data(self, location = '/Users/james/Desktop/sim data up_to_date/'):
        # workspace_vars = 'corner_a'+str(a_tool)+''
        # saved_data = np.load(workspace_vars+'.npz')

        # import file from the desktop
        workspace_vars = 'corner_a' + str(self.a_tool) + ''
        saved_data = np.load(location + workspace_vars + '.npz')

        Q_o, Q_com, P_p, t, steady_factor = saved_data['Qo_x'], saved_data['Qcom_x'], saved_data['Pp_x'], saved_data['time'], saved_data['steady_factor']

        self.sim_Q_out = np.interp(self.ts, t, Q_o)
        self.sim_Q_com = np.interp(self.ts, t, Q_com)
        self.sim_P_p = np.interp(self.ts, t, P_p)
        self.sim_ts = t
        self.sim_steady_factor = steady_factor

        return self.sim_Q_out, self.sim_Q_com, self.sim_P_p, self.sim_ts, self.sim_steady_factor

    def corner_swell_diameter(self, backflow_factor = 1.0, steady_factor = 1.0):

        # nominal_beadwidth = input_beadwidth[0]  # diameter of the nozzle [mm] (TYPE: 0.25)
        # backflow_factor = 1.0  # backflow correction factor
        # steady_factor = 1.03  # steady-state correction factor
        # ts = deepcopy(ts[:,0]) # [s]

        # accel_duration = v_tool / a_tool  # [s]
        accel_duration = self.precorner_decel_duration  # [s]
        swell_duration = np.sqrt(2 * (self.nominal_beadwidth / 2) / self.a_tool)  # [s] (TYP: np.sqrt(2*nozzle_diam/a_tool) )

        Q_out = self.Q_out[self.ts > 1] / backflow_factor  # [m^3/s]
        Q_com = self.Q_com[self.ts > 1] / steady_factor  # [m^3/s]
        W_com = self.W_com[self.ts > 1] * 1000  # diameters of the nozzle [mm] (TYPE: 0.25)
        ts = self.ts[self.ts > 1]  # [s]
        U_com = Q_com / (self.layer_height * W_com * 1e-09)  # [mm/s]

        t_B = ts[np.argmin(np.abs(Q_com - np.min(Q_com)))]  # [s]
        t_A = t_B - accel_duration  # [s]
        t_C = t_B + accel_duration  # [s]
        t_A_prime = t_B - swell_duration  # [s]
        t_C_prime = t_B + swell_duration  # [s]

        # np.seterr(divide='ignore')
        W_out = deepcopy(1e9 * Q_out / (U_com * self.layer_height))  # [mm]
        self.t_pre = deepcopy(ts[ts < t_A_prime])  # [s]
        self.w_pre = W_out[ts < t_A_prime]  # [mm]
        self.t_post = deepcopy(ts[ts > t_C_prime])  # [s]
        self.w_post = W_out[ts > t_C_prime]  # [mm]

        Q_out = Q_out[np.logical_and(ts >= t_B - 0.3, ts <= t_B + 0.7)]  # [m^3/s]
        Q_com = Q_com[np.logical_and(ts >= t_B - 0.3, ts <= t_B + 0.7)]  # [m^3/s]
        U_com = U_com[np.logical_and(ts >= t_B - 0.3, ts <= t_B + 0.7)]  # [mm/s]
        ts = ts[np.logical_and(ts >= t_B - 0.3, ts <= t_B + 0.7)]  # [s]

        # %%

        Q_o_A = Q_out[np.argmin(np.abs(ts - t_A))]  # [m^3/s]
        Q_o_B = Q_out[np.argmin(np.abs(ts - t_B))]  # [m^3/s]
        Q_o_C = Q_out[np.argmin(np.abs(ts - t_C))]  # [m^3/s]
        Q_o_A_prime = Q_out[np.argmin(np.abs(ts - t_A_prime))]  # [m^3/s]
        Q_o_C_prime = Q_out[np.argmin(np.abs(ts - t_C_prime))]  # [m^3/s]

        Q_com_A = Q_com[np.argmin(np.abs(ts - t_A))]  # [m^3/s]
        Q_com_B = Q_com[np.argmin(np.abs(ts - t_B))]  # [m^3/s]
        Q_com_C = Q_com[np.argmin(np.abs(ts - t_C))]  # [m^3/s]
        Q_com_A_prime = Q_com[np.argmin(np.abs(ts - t_A_prime))]  # [m^3/s]
        Q_com_C_prime = Q_com[np.argmin(np.abs(ts - t_C_prime))]  # [m^3/s]

        # dt = np.mean(ts[1:] - ts[:-1])

        D_swell = np.sqrt(
            4 * np.sum(Q_out[np.logical_and(ts >= t_A_prime, ts <= t_C_prime)] * 1e09 * self.dt) / (np.pi * self.layer_height)
        )

        self.ts_swell = ts
        self.Q_out_swell = Q_out
        self.Q_com_swell = Q_com

        self.t_A = t_A
        self.t_B = t_B
        self.t_C = t_C
        self.t_A_prime = t_A_prime
        self.t_C_prime = t_C_prime
        self.Q_com_A = Q_com_A
        self.Q_com_B = Q_com_B
        self.Q_com_C = Q_com_C
        self.Q_com_A_prime = Q_com_A_prime
        self.Q_com_C_prime = Q_com_C_prime
        self.Q_o_A = Q_o_A
        self.Q_o_B = Q_o_B
        self.Q_o_C = Q_o_C
        self.Q_o_A_prime = Q_o_A_prime
        self.Q_o_C_prime = Q_o_C_prime

        self.D_swell = D_swell


        return self.D_swell

    def analytical_error(self):

        error = self.sim_Q_out - self.Q_out
        self.error = error


    def sim_test_plots(self):
        fig_sim_test = plt.figure("sim test graph")
        ax = fig_sim_test.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Flowrate, [m$^3$/s]', fontdict=font)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts, self.sim_Q_com,
                 color='k', linewidth=2, linestyle='-',
                 label='commanded')

        plt.plot(self.ts, self.sim_Q_out,
                 color='r', linewidth=2, linestyle='-',
                 label='output')

    def test_plots(self):

        fig_test_1 = plt.figure("test 1 graph")
        ax = fig_test_1.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Commanded Flowrate, $Q_{com}$ [m$^3$/s]', fontdict=font)

        # plt.xlim(0,1)
        # plt.ylim(0,30)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts, self.Q_com,
                 color='k', linewidth=2, linestyle='-',
                 label='test')  # plot inlet (commanded) flow

        # ax.set_xticks(np.arange(0,1.1,0.1))
        # ax.set_yticks(np.arange(0,30,5))

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_test = plt.figure("test legend")
        # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])
        fig_test_2 = plt.figure("test 2 graph")
        ax = fig_test_2.add_subplot(1, 1, 1)
        plt.plot(self.ts, self.W_com)

    def corner_plots(self):

        fig_11 = plt.figure("test graph")
        ax = fig_11.add_subplot(1, 1, 1)

        plt.xlabel('Time [s]', fontdict=font)
        plt.ylabel('Volumetric Flowrate [mL/min]', fontdict=font)

        plt.xlim(0, 1)
        # plt.ylim(0,0.06)

        # plt.grid(which='major',visible=True,color='0.5',linestyle='-',linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts_swell - np.min(self.ts_swell), (self.Q_com_swell) / 1.6666667e-08,
                 color='k', linewidth=2, linestyle='-', zorder=0,
                 label='Input')  # plot inlet (commanded) flow

        plt.plot(self.ts_swell - np.min(self.ts_swell), (self.Q_out_swell) / 1.6666667e-08,
                 color='r', linewidth=2, zorder=1,
                 label='Output')  # plot outlet flow

        plt.scatter(x=np.array([self.t_A, self.t_B, self.t_C, self.t_A_prime, self.t_C_prime]) - np.min(self.ts_swell),
                    y=np.array([self.Q_com_A, self.Q_com_B, self.Q_com_C, self.Q_com_A_prime, self.Q_com_C_prime]) / (1.6666667e-08),
                    color='k', zorder=2)

        plt.scatter(x=np.array([self.t_A, self.t_B, self.t_C, self.t_A_prime, self.t_C_prime]) - np.min(self.ts_swell),
                    y=np.array([self.Q_o_A, self.Q_o_B, self.Q_o_C, self.Q_o_A_prime, self.Q_o_C_prime]) / (1.6666667e-08),
                    color='r', zorder=3)

        ax.yaxis.set_ticks(np.arange(0, 0.08, 0.02))
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))

        plt.gca().set_aspect('auto')  # (TYP: 15)

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_11 = plt.figure("flowrate legend")
        leg_11.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], ncol=2)

        # fig_11.savefig('corner_a'+str(a_tool)+'.png',bbox_inches='tight',dpi=600)
        # leg_11.savefig('corner_legend.png',bbox_inches='tight',dpi=600)

        plt.figure()

        plt.plot(self.t_pre, self.w_pre)
        plt.plot(self.t_post, self.w_post)

    def flow_bead_plots(self):
        flow_predictor_plots(
            ts=self.ts,
            Q_com = self.Q_com,
            W_com = self.W_com,
            Q_out = self.Q_out
        )


#%%
class corner_flowmatch:
    def __init__(self,
                 dt=0.01,  # time step [s]
                 a_tool=100,  # toolhead acceleration [mm/s^2]
                 v_tool=25,  # toolhead velocity [mm/s]
                 precorner_dist=180,  # straight-line distance before the corner [mm]
                 postcorner_dist=20,  # straight-line distance after the corner [mm]
                 layer_height=0.15,  # layer height [mm]
                 nominal_beadwidth=0.25,  # diameter of the nozzle [mm]
                 fluid = 'fluid_DOW121',
                 mixer = 'mixer_ISSM50nozzle',
                 pump = 'pump_viscotec_outdated'
                 ):
        # Parameters
        ##############################
        ##############################
        self.dt = dt  # time step [s]
        self.a_tool = a_tool  # toolhead acceleration [mm/s^2]
        self.v_tool = v_tool  # toolhead velocity [mm/s]
        self.precorner_dist = precorner_dist  # straight-line distance before the corner [mm]
        self.postcorner_dist = postcorner_dist  # straight-line distance after the corner [mm]
        self.layer_height = layer_height  # layer height [mm]
        self.nominal_beadwidth = nominal_beadwidth  # diameter of the nozzle [mm]
        self.fluid = fluid
        self.mixer = mixer
        self.pump = pump

    def pathgen(self):

        acceleration_dist = (self.v_tool ** 2) / (2 * self.a_tool)  # distance traveled while decelerating/accelerating between v_tool and 0mm/s [mm]

        self.precorner_steady_duration = (self.precorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]
        self.precorner_decel_duration = self.v_tool / self.a_tool  # duration of decreasing toolhead velocity [s]
        self.postcorner_accel_duration = self.v_tool / self.a_tool  # duration of increasing toolhead velocity [s]
        self.postcorner_steady_duration = (self.postcorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]

        self.duration = (self.precorner_steady_duration
                    + self.precorner_decel_duration
                    + self.postcorner_steady_duration
                    + self.postcorner_accel_duration)  # [s]

        # %%

        t = np.arange(0, self.duration + self.dt, self.dt)  # initialize list of time coordinates [s]

        V_com = (self.v_tool) * np.ones(np.shape(t))  # commanded toolhead velocity: steady pre-corner [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration
            )] = np.linspace(self.v_tool, 0, np.sum(np.logical_and(
                    t >= self.precorner_steady_duration,
                    t <= self.precorner_steady_duration + self.precorner_decel_duration
                    )))  # commanded toolhead velocity: transient pre-corner (decelerating) [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration + self.precorner_decel_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
            )] = np.linspace(0, self.v_tool, np.sum(np.logical_and(
                t >= self.precorner_steady_duration + self.precorner_decel_duration,
                t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
                )))  # commanded toolhead velocity: transient post-corner (accelerating) [mm/s]

        Q_com = V_com * self.layer_height * self.nominal_beadwidth * 1e-09
        Q_com[t < self.precorner_steady_duration / 16] = 1 * 1.6666667e-08  # flow rate: 1 mL/min, converted to [m^3/s]
        Q_com[t < 0.001] = 0 * 1e-09

        input_W = self.nominal_beadwidth * np.ones(np.shape(t)) / 1000  # diameters of the nozzle [m]
        input_W[t > self.precorner_steady_duration / 16] = np.max(input_W[t > self.precorner_steady_duration / 16]) * Q_com[
            t > self.precorner_steady_duration / 16] / np.max(
            Q_com[t > self.precorner_steady_duration / 16])  # diameters of the nozzle [m]

        print('Simulated duration: ' + str(round(self.duration, 2)) + ' seconds.')

        self.input_t = t
        self.input_Q = Q_com
        self.input_W = input_W

        return self.input_t, self.input_Q, self.input_W

    def flow_predictor(self):
        ts, W_com, Q_com, Q_out = flow_predictor(
            ts=self.input_t,
            input_flowrate=self.input_Q,
            input_beadwidth=self.input_W,
            fluid = self.fluid,
            mixer = self.mixer,
            pump = self.pump
        )

        self.ts = ts
        self.W_com = W_com
        self.Q_com = Q_com
        self.Q_out = Q_out

        return self.ts, self.W_com, self.Q_com, self.Q_out

    def test_plots(self):

        fig_test_1 = plt.figure("test 1 graph")
        ax = fig_test_1.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Commanded Flowrate, $Q_{com}$ [m$^3$/s]', fontdict=font)

        # plt.xlim(0,1)
        # plt.ylim(0,30)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts, self.Q_com,
                 color='k', linewidth=2, linestyle='-',
                 label='test')  # plot inlet (commanded) flow

        # ax.set_xticks(np.arange(0,1.1,0.1))
        # ax.set_yticks(np.arange(0,30,5))

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_test = plt.figure("test legend")
        # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])
        fig_test_2 = plt.figure("test 2 graph")
        ax = fig_test_2.add_subplot(1, 1, 1)
        plt.plot(self.ts, self.W_com)

    def corner_plots(self):

        t_B = self.precorner_steady_duration + self.precorner_decel_duration  # [s]
        t_A = t_B - self.precorner_decel_duration  # [s]
        t_C = t_B + self.postcorner_accel_duration  # [s]
        print(t_A, t_B, t_C)

        fig_11 = plt.figure("test graph")
        ax = fig_11.add_subplot(1, 1, 1)

        plt.xlabel('Time [s]', fontdict=font)
        plt.ylabel('Volumetric Flowrate [mL/min]', fontdict=font)

        plt.xlim(t_A, t_C)
        plt.ylim(0,0.10)

        # plt.grid(which='major',visible=True,color='0.5',linestyle='-',linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts - np.min(self.ts), (self.Q_com) / 1.6666667e-08,
                 color='k', linewidth=2, linestyle='-', zorder=0,
                 label='Input')  # plot inlet (commanded) flow

        plt.plot(self.ts - np.min(self.ts), (self.Q_out) / 1.6666667e-08,
                 color='r', linewidth=2, zorder=1,
                 label='Output')  # plot outlet flow

        # plt.scatter(x=np.array([self.t_A, self.t_B, self.t_C, self.t_A_prime, self.t_C_prime]) - np.min(self.ts_swell),
        #             y=np.array([self.Q_com_A, self.Q_com_B, self.Q_com_C, self.Q_com_A_prime, self.Q_com_C_prime]) / (1.6666667e-08),
        #             color='k', zorder=2)
        #
        # plt.scatter(x=np.array([self.t_A, self.t_B, self.t_C, self.t_A_prime, self.t_C_prime]) - np.min(self.ts_swell),
        #             y=np.array([self.Q_o_A, self.Q_o_B, self.Q_o_C, self.Q_o_A_prime, self.Q_o_C_prime]) / (1.6666667e-08),
        #             color='r', zorder=3)

        # ax.yaxis.set_ticks(np.arange(0, 0.08, 0.02))
        # ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))

        plt.gca().set_aspect('auto')  # (TYP: 15)

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        leg_11 = plt.figure("flowrate legend")
        leg_11.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], ncol=2)

        # fig_11.savefig('corner_a'+str(a_tool)+'.png',bbox_inches='tight',dpi=600)
        # leg_11.savefig('corner_legend.png',bbox_inches='tight',dpi=600)


    def flow_bead_plots(self):
        flow_predictor_plots(
            ts=self.ts,
            Q_com = self.Q_com,
            W_com = self.W_com,
            Q_out = self.Q_out
        )


#%%
class twostep:
    def __init__(self,
                 dt=0.001,  # time step [s]
                 tmax=50,  # maximum simulation time [s]
                 flowrate_up=1,  # step-up inlet flow rate [mL/min]
                 flowrate_down=0.001,  # step-down inlet flow rate [mL/min]
                 t_flowup=0.0001,  # time of step-up command [s]
                 t_flowdown=7.5,  # time of step-down command [s]
                 beadwidth_down=0.0001,  # step-down bead width [m]
                 beadwidth_up=0.0029,  # step-up bead width [m]
                 t_beaddown=10,  # time of step-up command [s]
                 t_beadup=11,  # time of step-down command [s]
                 fluid = 'fluid_DOW121',
                 mixer = 'mixer_ISSM50nozzle',
                 pump = 'pump_viscotec_outdated'
                 ):
        # Parameters
        ##############################
        ##############################
        self.dt = dt  # time step [s]
        self.tmax = tmax  # maximum simulation time [s]
        self.flowrate_up = flowrate_up  # step-up inlet flow rate [mL/min]
        self.flowrate_down = flowrate_down  # step-down inlet flow rate [mL/min]
        self.t_flowup = t_flowup  # time of step-up command [s]
        self.t_flowdown = t_flowdown  # time of step-down command [s]
        self.beadwidth_down = beadwidth_down  # step-down bead width [m]
        self.beadwidth_up = beadwidth_up  # step-up bead width [m]
        self.t_beaddown = t_beaddown  # time of step-up command [s]
        self.t_beadup = t_beadup  # time of step-down command [s]
        self.fluid = fluid
        self.mixer = mixer
        self.pump = pump


    def pathgen(self):

        ts = np.arange(0, self.tmax, self.dt)  # initialize list of time coordinates

        input_Q = np.ones(np.shape(ts)) * self.flowrate_down / 6e7  # inlet flow rate: 0 mL/min at t=0
        input_Q[ts > self.t_flowup] = self.flowrate_up / 6e7  # inlet flow rate: 1 mL/min at t=0.0001 (step up), converted to [m^3/s]
        input_Q[ts > self.t_flowdown] = self.flowrate_down / 6e7  # inlet flow rate: 0 mL/min at t=10 (step down), converted to [m^3/s]

        input_W = self.beadwidth_up * np.ones(np.shape(ts))
        input_W[ts > self.t_beaddown] = self.beadwidth_down
        input_W[ts > self.t_beadup] = self.beadwidth_up

        print('Simulated duration: ' + str(round(self.tmax,2)) + ' seconds.')

        self.input_t = ts
        self.input_Q = input_Q
        self.input_W = input_W

        return self.input_t, self.input_Q, self.input_W

    def flow_predictor(self):
        ts, W_com, Q_com, Q_out = flow_predictor(
            ts=self.input_t,
            input_flowrate=self.input_Q,
            input_beadwidth=self.input_W,
            fluid = self.fluid,
            mixer = self.mixer,
            pump = self.pump
        )

        self.ts = ts
        self.W_com = W_com
        self.Q_com = Q_com
        self.Q_out = Q_out

        return self.ts, self.W_com, self.Q_com, self.Q_out

    def test_plots(self):


        fig_test = plt.figure("test graph")
        ax = fig_test.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Commanded Flowrate, $Q_{com}$ [m^3/s]', fontdict=font)

        # plt.xlim(0,1)
        # plt.ylim(0,30)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.ts, self.Q_com,
                 color='k', linewidth=2, linestyle='-',
                 label='test')  # plot inlet (commanded) flow

        # ax.set_xticks(np.arange(0,1.1,0.1))
        # ax.set_yticks(np.arange(0,30,5))

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_test = plt.figure("test legend")
        # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])

    def linewidths(self,
                   v_tool=25,  # toolhead velocity [mm/s]
                   layer_height = 0.15  # layer height [mm]
                   ):

        self.linewidths = 1000 * self.Q_out / ((v_tool / 1000) * (layer_height / 1000))

        return self.linewidths

    def linewidths_plots(self):

        fig_linewidths = plt.figure("linewidth graph")
        ax = fig_linewidths.add_subplot(1, 1, 1)

        plt.plot(self.ts, self.linewidths / 2,
                 color='r', linewidth=2)
        plt.plot(self.ts, self.linewidths / -2,
                 color='r', linewidth=2)

    def flow_bead_plots(self):
        flow_predictor_plots(
            ts=self.ts,
            Q_com = self.Q_com,
            W_com = self.W_com,
            Q_out = self.Q_out
        )


class training_corner:
    def __init__(self,
                 a_tool,  # toolhead acceleration [mm/s^2]
                 v_tool,  # toolhead velocity [mm/s]
                 precorner_dist = 180,  # straight-line distance before the corner [mm]
                 postcorner_dist = 20,  # straight-line distance after the corner [mm]
                 pulse_magnitude = 1, # pulse magnitude [mL/min]
                 pulse_dists = [50,60],  # points during the precorner to apply the stepup and stepdown [mm]
                 layer_height = 0.15,  # layer height [mm]
                 nominal_beadwidth = 0.25,  # diameter of the nozzle [mm],
                 steady_factor = 1.00,  # factor by which to multiply the steady-state flow rate (TYP: 1.00)
                 dt=4 * 1e-5 / 880,  # time step [s]
                 fluid = 'fluid_DOW121',
                 mixer = 'mixer_ISSM50nozzle',
                 pump = 'pump_viscotec_outdated'
                 ):
        # Parameters
        ##############################
        ##############################
        self.dt = dt  # time step [s]
        self.a_tool = a_tool  # toolhead acceleration [mm/s^2]
        self.v_tool = v_tool  # toolhead velocity [mm/s]
        self.precorner_dist = precorner_dist  # straight-line distance before the corner [mm]
        self.postcorner_dist = postcorner_dist  # straight-line distance after the corner [mm]
        self.pulse_magnitude = pulse_magnitude  # pulse magnitude [mL/min]
        self.pulse_dists = pulse_dists  # points during the precorner to apply the stepup and stepdown [mm]
        self.layer_height = layer_height  # layer height [mm]
        self.nominal_beadwidth = nominal_beadwidth  # diameter of the nozzle [mm]
        self.steady_factor = steady_factor # factor by which to multiply the steady-state flow rate (TYP: 1.00)
        self.fluid = fluid
        self.mixer = mixer
        self.pump = pump




    def pathgen(self):

        acceleration_dist = (self.v_tool ** 2) / (2 * self.a_tool)  # distance traveled while decelerating/accelerating between v_tool and 0mm/s [mm]

        self.precorner_steady_duration = (self.precorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]
        self.precorner_decel_duration = self.v_tool / self.a_tool  # duration of decreasing toolhead velocity [s]
        self.postcorner_accel_duration = self.v_tool / self.a_tool  # duration of increasing toolhead velocity [s]
        self.postcorner_steady_duration = (self.postcorner_dist - acceleration_dist) / self.v_tool  # duration of steady toolhead velocity [s]

        self.duration = (self.precorner_steady_duration
                    + self.precorner_decel_duration
                    + self.postcorner_steady_duration
                    + self.postcorner_accel_duration)  # [s]

        # %%

        t = np.arange(0, self.duration + self.dt, self.dt)  # initialize list of time coordinates [s]

        V_com = (self.v_tool) * np.ones(np.shape(t))  # commanded toolhead velocity: steady pre-corner [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration
            )] = np.linspace(self.v_tool, 0, np.sum(np.logical_and(
                    t >= self.precorner_steady_duration,
                    t <= self.precorner_steady_duration + self.precorner_decel_duration
                    )))  # commanded toolhead velocity: transient pre-corner (decelerating) [mm/s]
        V_com[np.logical_and(
            t >= self.precorner_steady_duration + self.precorner_decel_duration,
            t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
            )] = np.linspace(0, self.v_tool, np.sum(np.logical_and(
                t >= self.precorner_steady_duration + self.precorner_decel_duration,
                t <= self.precorner_steady_duration + self.precorner_decel_duration + self.postcorner_accel_duration
                )))  # commanded toolhead velocity: transient post-corner (accelerating) [mm/s]

        Q_com = V_com * self.layer_height * self.nominal_beadwidth * 1e-09
        Q_com[t < self.precorner_steady_duration / 16] = self.pulse_magnitude * 1.6666667e-08  # flow rate: 1 mL/min, converted to [m^3/s]]
        Q_com[t < 0.001] = 0 * 1e-09
        Q_com[np.logical_and(t > (self.pulse_dists[0]-acceleration_dist)/self.v_tool,
                             t < (self.pulse_dists[1]-acceleration_dist)/self.v_tool)] = self.pulse_magnitude * 1.6666667e-08

        input_W = self.nominal_beadwidth * np.ones(np.shape(t)) / 1000  # diameters of the nozzle [m]
        # input_W[t > precorner_steady_duration / 16] = np.max(input_W[t > precorner_steady_duration / 16]) * Q_com[
        #     t > precorner_steady_duration / 16] / np.max(
        #     Q_com[t > precorner_steady_duration / 16])  # diameters of the nozzle [m]

        print('Simulated duration: ' + str(round(self.duration, 2)) + ' seconds.')

        self.input_t = t
        self.input_Q = Q_com*self.steady_factor
        self.input_W = input_W

        return self.input_t, self.input_Q, self.input_W


    def test_plots(self):

        fig_test = plt.figure("test graph")
        ax = fig_test.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Commanded Flowrate, $Q_{com}$ [m^3/s]', fontdict=font)

        # plt.xlim(0,1)
        # plt.ylim(0,30)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.input_t, self.input_Q,
                 color='k', linewidth=2, linestyle='-',
                 label='test')  # plot inlet (commanded) flow

        # ax.set_xticks(np.arange(0,1.1,0.1))
        # ax.set_yticks(np.arange(0,30,5))

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_test = plt.figure("test legend")
        # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])


class training_twosteps:
    def __init__(self,
                 flowrate_magnitudes,  # step-up and step-down inlet flow rates [mL/min]
                 flowrate_times,  # times of step-up command [s]
                 beadwidth_magnitudes,  # step-up and step-down bead widths [m]
                 beadwidth_times,  # times of beadwidth command [s]
                 flowrate_start = 0.001,  # initial inlet flow rate [mL/min]
                 beadwidth_start = 0.0029,  # initial bead width [m]
                 dt=4*1e-5/880,  # time step [s]
                 tmax=8,  # maximum simulation time [s]
                 fluid = 'fluid_DOW121',
                 mixer = 'mixer_ISSM50nozzle',
                 pump = 'pump_viscotec_outdated'
                 ):
        # Parameters
        ##############################
        ##############################
        self.flowrate_magnitudes = flowrate_magnitudes
        self.flowrate_times = flowrate_times
        self.beadwidth_magnitudes = beadwidth_magnitudes
        self.beadwidth_times = beadwidth_times
        self.flowrate_start = flowrate_start
        self.beadwidth_start = beadwidth_start
        self.dt = dt  # time step [s]
        self.tmax = tmax  # maximum simulation time [s]
        self.fluid = fluid
        self.mixer = mixer
        self.pump = pump

    def pathgen(self):

        ts = np.arange(0, self.tmax, self.dt)  # initialize list of time coordinates

        input_Q = np.ones(np.shape(ts)) * self.flowrate_start / 6e7  # inlet flow rate: 0 [m3/s]
        for i in range(len(self.flowrate_times)):
            input_Q[ts > self.flowrate_times[i][0]] = self.flowrate_magnitudes[i][0] / 6e7
            input_Q[ts > self.flowrate_times[i][1]] = self.flowrate_magnitudes[i][1] / 6e7

        input_W = np.ones(np.shape(ts)) * self.beadwidth_start  # bead width: 0.0029 [m]
        for i in range(len(self.beadwidth_times)-1):
            input_W[ts > self.beadwidth_times[i][0]] = self.beadwidth_magnitudes[i][0]
            input_W[ts > self.beadwidth_times[i][1]] = self.beadwidth_magnitudes[i][1]

        print('Simulated duration: ' + str(round(self.tmax,2)) + ' seconds.')

        self.input_t = ts
        self.input_Q = input_Q
        self.input_W = input_W

        return self.input_t, self.input_Q, self.input_W

    def test_plots(self):

        fig_test = plt.figure("test graph")
        ax = fig_test.add_subplot(1, 1, 1)

        plt.xlabel('Time, $t$ [s]', fontdict=font)
        plt.ylabel('Commanded Flowrate, $Q_{com}$ [m^3/s]', fontdict=font)

        # plt.xlim(0,1)
        # plt.ylim(0,30)

        plt.grid(which='major', visible=True, color='0.5', linestyle='-', linewidth=0.5)

        plt.xscale('linear')
        plt.yscale('linear')

        plt.plot(self.input_t, self.input_Q,
                 color='k', linewidth=2, linestyle='-',
                 label='test')  # plot inlet (commanded) flow

        # ax.set_xticks(np.arange(0,1.1,0.1))
        # ax.set_yticks(np.arange(0,30,5))

        # ax.xaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useMathText=True)

        # ax.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)

        # ax.legend()

        # leg_test = plt.figure("test legend")
        # leg_test.legend(ax.get_legend_handles_labels()[0],ax.get_legend_handles_labels()[1])





if __name__ == '__main__':
    test1 = training_twosteps(
        flowrate_magnitudes = [[1, 0.001], [2, 0.001], [3, 0.001]],
        flowrate_times = [[0.0001, 5], [15, 20], [30, 35]],
        beadwidth_magnitudes=[[0.0001, 0.0029]],
        beadwidth_times= [[10, 11]]
    )
    test1.pathgen()
    test1.test_plots()

    test2 = training_corner(
        a_tool=100,
        v_tool=55,
        precorner_dist=180,
        postcorner_dist=50,
        pulse_magnitude=1,
        pulse_dists=[50, 70],
        layer_height=0.15,
        nominal_beadwidth=0.25,
    )
    test2.pathgen()
    test2.test_plots()
