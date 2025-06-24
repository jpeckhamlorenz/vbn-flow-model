import matplotlib.pyplot as plt
from pathgen import training_twosteps, training_corner, corner_naive, corner_flowmatch
from CM_sim import flow_simulator
import numpy as np
plt.close('all')
import itertools
import torch
from constants.filepath import PROJECT_PATH
import os

# if torch.backends.mps.is_available():
#     mps_device = torch.device("mps")
#     # x = torch.ones(1, device=mps_device)
#     # print(x)
# else:
#     print("MPS device not found.")

plt.close('all')

# from flow_predictor import flow_predictor
from flow_predictor_hybrid import flow_predictor_lstm

#%%

# from traj_WALR import LightningModule, get_best_run
# run_id, run_config = get_best_run()
# checkpoint_id = os.listdir(os.path.join(PROJECT_PATH, 'VBN-modeling', run_id, 'checkpoints'))[0]  # Assuming only one ckpt file exists
# checkpoint = torch.load(
#     os.path.join(PROJECT_PATH, 'VBN-modeling', run_id, 'checkpoints', checkpoint_id),
#     map_location=torch.device('cpu')  # Load to CPU first
# )

# model = LightningModule.load_from_checkpoint(
#         os.path.join(PROJECT_PATH, 'VBN-modeling', run_id, 'checkpoints', checkpoint_id),
#         config=run_config)
# model.eval()



#%%


# import torch
# from traj_WALR import WalrLSTM as LSTMmodel
# from traj_WALR import get_best_run
#
#
#
# model = LSTMmodel(hidden_size = run_config.hidden_size, num_layers = run_config.num_layers).to('mps')
#
# state_dict = checkpoint['state_dict']
# remove_prefix = 'net.'
# state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
# model.load_state_dict(state_dict)
#
# model.eval()


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

corner_flowmatch_test.pathgen()
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





# twosteps = []

# for a,b,c in itertools.permutations([1,2,3]):
#         for d, e, f, g, h in itertools.permutations([0,1,2,3,4]):
#                 twosteps.append(training_twosteps(
#                         flowrate_magnitudes = [[a, 0.001], [b, 0.001], [c, 0.001]],
#                         flowrate_times = [[0.0001, 5+d], [15+e, 20+f], [30+g, 35+h]],
#                         beadwidth_magnitudes = [[]],
#                         beadwidth_times = [[]]
#                 ))

#%%

# for d, e, f in itertools.permutations([0.0,0.5,1.0]):
#         twostep = training_twosteps(
#                 flowrate_magnitudes = [[1.0, 0.001], [0.5, 0.001]],
#                 flowrate_times = [[0.0001, 2+d], [4+e, 6+f]],
#                 beadwidth_magnitudes = [[]],
#                 beadwidth_times = [[]]
#         )
#         t, Q, W = twostep.pathgen()
#         del twostep
#         flow_simulator(t,Q, file_name = 'twostep_21_' + str([d,e,f]))



#%%


# tool_accelerations = [100, 250, 500, 1000, 1500, 2000]
#
#
# for acceleration in tool_accelerations:
#         corner = training_corner(
#                 a_tool=acceleration,
#                 v_tool=20,
#                 precorner_dist=200,
#                 postcorner_dist=70,
#                 pulse_magnitude=0.75,
#                 pulse_dists=[60, 100],
#                 layer_height=0.15,
#                 nominal_beadwidth=0.25
#         )
#         t, Q, W = corner.pathgen()
#         del corner
#         flow_simulator(t,Q, file_name = 'corner_60100_' + str(acceleration))


#%%

# test = training_twosteps(flowrate_magnitudes = [[0.75, 0.001], [0.1, 0.001], [0.86, 0.001]],
#                          flowrate_times = [[0.0001, 0.01], [17, 18], [19.5, 33]],
#                          beadwidth_magnitudes = [[]],
#                          beadwidth_times = [[]],
#                          dt = 0.001,
#                          tmax=40)
#
# # test = training_corner(
# #         a_tool=100,
# #         v_tool=15,
# #         precorner_dist=210,
# #         postcorner_dist=90,
# #         pulse_magnitude=0.696,
# #         pulse_dists=[55, 72],
# #         layer_height=0.15,
# #         nominal_beadwidth=0.25,
# #         dt = 0.001,
# # )
#
# # test = corner_flowmatch(
# #         dt = 0.001,  # time step [s]
# #         a_tool = 100,  # toolhead acceleration [mm/s^2]
# #         v_tool = 25,  # toolhead velocity [mm/s]
# #         precorner_dist = 180,  # straight-line distance before the corner [mm] (TYP: 180)
# #         postcorner_dist = 30,  # straight-line distance after the corner [mm]  (TYP: 30)
# #         layer_height = 0.15,  # layer height [mm]
# #         nominal_beadwidth = 0.3,  # diameter of the nozzle [mm]
# #         pump = 'pump_viscotec_outdated'
# # )
#
#
# test_t_input, test_Q_input, test_W_input = test.pathgen()
#
#
#
# test_t, test_W, test_Q_com, test_Q_out = flow_predictor(test_t_input, test_Q_input, test_W_input)
#
#
# IC_cutoff = 0.2
# test_W = test_W[test_t > IC_cutoff]
# test_Q_com = test_Q_com[test_t > IC_cutoff]
# test_Q_out = test_Q_out[test_t > IC_cutoff]
# test_t = test_t[test_t > IC_cutoff]
#
#
#
#
# test_analytical = test_Q_out * 1e9
# test_command = test_Q_com * 1e9
# test_combined = np.stack((test_command, test_analytical), axis=1)
# test_input = torch.tensor(test_combined, dtype=torch.float32, device=mps_device)
#
#
#
# # class ResidualModel(torch.nn.Module):
# #     def __init__(self):
# #         super(ResidualModel, self).__init__()
# #         # self.fc1 = torch.nn.Linear(2, 64)  # Input layer
# #         # self.fc2 = torch.nn.Linear(64, 256)  # Hidden layer
# #         # self.fc3 = torch.nn.Linear(256, 64)  # Output layer
# #         # self.fc4 = torch.nn.Linear(64, 1)  # Output layer
# #
# #         self.fc1 = torch.nn.Linear(2, 64)  # Input layer
# #         self.fc2 = torch.nn.Linear(64, 512)  # Hidden layer
# #         self.fc3 = torch.nn.Linear(512, 1024)  # Hidden layer
# #         self.fc4 = torch.nn.Linear(1024, 512)  # Output layer
# #         self.fc5 = torch.nn.Linear(512, 64)  # Output layer
# #         self.fc6 = torch.nn.Linear(64, 1)  # Output layer
# #
# #     def forward(self, x):
# #         x = torch.relu(self.fc1(x))
# #         x = torch.relu(self.fc2(x))
# #         x = torch.relu(self.fc3(x))
# #         x = torch.relu(self.fc4(x))
# #         x = torch.relu(self.fc5(x))
# #         return self.fc6(x)
# #
# #     def traj_correction(self, traj_input):
# #         self.load_state_dict(torch.load('trajectory_correction_DNN_state_v1.pth', weights_only = True))
# #         self.eval()
# #         with torch.no_grad():
# #             output = model(traj_input) * 1e-9
# #         Q_out_correction = (output).cpu().numpy().reshape(-1)
# #         Q_out_total = test_Q_out + Q_out_correction
# #
# #         return Q_out_total, Q_out_correction
#
#
# model = ResidualModel()
# model.to(mps_device)
# Q_out_total, test_result = model.traj_correction(test_input, test_Q_out)
#
# # model.eval()
# #
# # with torch.no_grad():
# #     output = model(test_input) * 1e-9
# #
# #
# # test_result = (output).cpu().numpy().reshape(-1)
# # Q_out_total = test_Q_out + test_result
#
#
#
# plt.figure()
# plt.plot(test_t, test_Q_com, label='Input', color='black', linestyle='--')
# # plt.plot(test_accel.ts, test_accel.sim_Q_out, label = 'Simulated Data', color = 'red')
# plt.plot(test_t, test_Q_out, label = 'Analytical Data', color = 'blue')
# plt.plot(test_t, test_result, label='Residual', color='magenta')
# plt.plot(test_t, Q_out_total, label='Total', color='red')
#
# plt.legend()


#%%

# test = training_twosteps(flowrate_magnitudes=[[0.2, 0.001], [0.48, 0.001]],
#                          flowrate_times=[[0.0001, 4], [7, 16]],
#                          beadwidth_magnitudes=[[]],
#                          beadwidth_times=[[]],
#                          dt=0.001,
#                          tmax=20)
time, command, bead = corner_flowmatch_test.pathgen()

# prediction, residual, analytical = flow_predictor_lstm(time, command, bead,
#                                                         flowrate_regularization=1e9,
#                                                         model_filename='lstm_residual_model_v7.pth')

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