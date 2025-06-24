import matplotlib.pyplot as plt
from inputs.pathgen import corner_flowmatch

plt.close('all')
from flow_predictor_lstm import flow_predictor_lstm


#%%

# corner_naive_test = corner_naive(
#         dt = 0.001,  # time step [s]
#         a_tool = 100,  # toolhead acceleration [mm/s^2]
#         v_tool = 25,  # toolhead velocity [mm/s]
#         precorner_dist = 100,  # straight-line distance before the corner [mm] (TYP: 180)
#         postcorner_dist = 30,  # straight-line distance after the corner [mm]  (TYP: 30)
#         layer_height = 0.15,  # layer height [mm]
#         nominal_beadwidth = 0.25,  # diameter of the nozzle [mm]
#         steady_factor = 1.00,  # factor by which to multiply the steady-state flow rate (TYP: 1.00)
#         pump = 'pump_viscotec_outdated'
# )
#
# corner_naive_test.pathgen()
# corner_naive_test.flow_predictor()
# corner_naive_test.load_sim_data()
# corner_naive_test.corner_swell_diameter()
# corner_naive_test.sim_test_plots()
# corner_naive_test.test_plots()
# corner_naive_test.corner_plots()
# corner_naive_test.flow_bead_plots()



#%%

corner_flowmatch_test = corner_flowmatch(
        dt = 0.001,  # time step [s]
        a_tool = 100,  # toolhead acceleration [mm/s^2]
        v_tool = 25,  # toolhead velocity [mm/s]
        precorner_dist = 180,  # straight-line distance before the corner [mm] (TYP: 180)
        postcorner_dist = 30,  # straight-line distance after the corner [mm]  (TYP: 30)
        layer_height = 0.15,  # layer height [mm]
        nominal_beadwidth = 0.3,  # diameter of the nozzle [mm]
        pump = 'pump_viscotec_outdated'
)

# corner_flowmatch_test.pathgen()
# corner_flowmatch_test.flow_predictor()
# corner_flowmatch_test.test_plots()
# corner_flowmatch_test.corner_plots()
# corner_flowmatch_test.flow_bead_plots()


#%%

# twostep_test = twostep(
#         dt = 0.1,  # time step [s]
#         tmax = 40,  # maximum simulation time [s]
#         flowrate_up = 0.7,  # step-up inlet flow rate [mL/min]
#         flowrate_down = 0.001,  # step-down inlet flow rate [mL/min]
#         t_flowup = 5,  # time of step-up command [s] (TYP: 0.001)
#         t_flowdown = 23,  # time of step-down command [s]
#         beadwidth_down = 0.0001,  # step-down bead width [m]
#         beadwidth_up = 0.0029, # step-up bead width [m]
#         t_beaddown = 41,  # time of step-up command [s]
#         t_beadup = 42,  # time of step-down command [s]
# )
#
# twostep_test.pathgen()
# twostep_test.flow_predictor()
# twostep_test.test_plots()
# twostep_test.linewidths()
# twostep_test.linewidths_plots()
# twostep_test.flow_bead_plots()



#%%

# test = training_twosteps(flowrate_magnitudes=[[0.2, 0.001], [0.48, 0.001]],
#                          flowrate_times=[[0.0001, 4], [7, 16]],
#                          beadwidth_magnitudes=[[]],
#                          beadwidth_times=[[]],
#                          dt=0.001,
#                          tmax=20)
time, command, bead = corner_flowmatch_test.pathgen()


prediction, analytical = flow_predictor_lstm(time, command, bead, 'WALR')

# plot results
plot_skip = 0.2  # Adjust this to skip more points if needed
test = plt.figure(figsize=(12, 6))
plt.plot(time[time > plot_skip], command[time > plot_skip], label='Commanded Flow Rate', color='black', linestyle='--')
plt.plot(time[time > plot_skip], analytical[time > plot_skip], label='Analytical Flow Rate', color='blue')
# plt.plot(time[time > plot_skip], residual[time > plot_skip], label='Residual Prediction', color='magenta')
plt.plot(time[time > plot_skip], prediction[time > plot_skip], label='Total Flow Rate', color='green')

plt.xlabel('Time (s)')
plt.ylabel('Flow Rate (m3/s)')
plt.title('Flow Rate Prediction')
plt.legend()
plt.grid()

# save the figure at 600 dpi

# test.savefig('corner_flowmatching_LSTM_test.png', bbox_inches='tight', dpi=600)