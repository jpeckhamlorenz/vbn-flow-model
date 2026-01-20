import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.close("all")

from inputs.pathgen import corner_flowmatch, training_twosteps, twostep
from flow_predictor_lstm import flow_predictor_lstm_windowed, WindowParams

# test = twostep(
#         dt = 0.1,  # time step [s]
#         tmax = 20,  # maximum simulation time [s]
#         flowrate_up = 0.7,  # step-up inlet flow rate [mL/min]
#         flowrate_down = 0.001,  # step-down inlet flow rate [mL/min]
#         t_flowup = 5,  # time of step-up command [s] (TYP: 0.001)
#         t_flowdown = 17,  # time of step-down command [s]
#         beadwidth_down = 0.0008,  # step-down bead width [m]
#         beadwidth_up = 0.0029, # step-up bead width [m]
#         t_beaddown = 50,  # time of step-up command [s]
#         t_beadup = 170,  # time of step-down command [s]
# )


# test = training_twosteps(flowrate_magnitudes=[[0.2, 0.001], [0.48, 0.001]],
#                          flowrate_times=[[0.0001, 4], [7, 16]],
#                          beadwidth_magnitudes=[[]],
#                          beadwidth_times=[[]],
#                          dt=0.001,
#                          tmax=20)

test = corner_flowmatch(
    dt=0.001,  # time step [s]
    a_tool=100,  # toolhead acceleration [mm/s^2]
    v_tool=25,  # toolhead velocity [mm/s]
    precorner_dist=180,  # straight-line distance before the corner [mm] (TYP: 180)
    postcorner_dist=30,  # straight-line distance after the corner [mm]  (TYP: 30)
    layer_height=0.15,  # layer height [mm]
    nominal_beadwidth=0.9,  # diameter of the nozzle [mm]
    pump='pump_viscotec_outdated'
)

time, command, bead = test.pathgen()
time = np.asarray(time)
command = np.asarray(command)
bead = np.asarray(bead)

# # --- choose your trained checkpoint
# ckpt_path = Path("traj_WALR-test/lightning_logs/version_6/checkpoints/last.ckpt")
#
# # --- must match model init signature used in training
# config = {
#     "lr": 0.001,
#     "num_layers": 2,
#     "hidden_size": 128,
#     "batch_size": 64,
#     "huber_delta": 1.0,
# }
# run_config = DictToObject(config)

from models.traj_WALR import DataModule, DictToObject, get_best_run
run_id, run_config = get_best_run(sweep_id='mnywg829')

ckpt_dir = Path('./VBN-modeling') / run_id / 'checkpoints'
ckpt_path = sorted(ckpt_dir.glob('*.ckpt'))[-1].absolute()

# NOTE: point these to your real dataset/splits used during training
data = DataModule(
    run_config,
    data_folderpath=Path("./dataset/recursive_samples")
)

# Your updated DataModule should populate norm_stats here
if hasattr(data, "setup"):
    data.setup("fit")

if not hasattr(data, "norm_stats"):
    raise AttributeError(
        "DataModule has no attribute norm_stats. "
        "Make sure you're using the latest traj_WALR DataModule that computes/stores norm_stats."
    )

norm_stats = data.norm_stats

# --- windowing parameters (choose freely)
win = WindowParams(
    window_len_s=4.5,
    window_step_s=0.1,
    include_tail=True,
)

Q_pred, Q_vbn, Q_res_pred = flow_predictor_lstm_windowed(
    time_np=time,
    command_np=command,
    bead_np=bead,
    model_type="WALR",
    ckpt_path=ckpt_path,
    run_config=run_config,
    norm_stats=norm_stats,
    win=win,
    device_type="mps",
    flow_scale=1e9,
    bead_units="m",   # set "mm" if bead is already mm
)

# --- plots
plot_skip = 0.2
mask = time > plot_skip

plt.figure(figsize=(12, 5))
plt.plot(time[mask], command[mask] * 1e9, "--", label="Command (mm^3/s)")
# plt.plot(time[mask], Q_vbn[mask] * 1e9, label="Analytical VBN (mm^3/s)")
plt.plot(time[mask], Q_pred[mask] * 1e9, label="VBN + LSTM (mm^3/s)")
# plt.plot(time[mask], Q_res_pred[mask] * 1e9, label="LSTM (mm^3/s)")
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Flow [mm^3/s]")
plt.title("Deployed flow predictor (windowed + stitched, normalized)")
plt.show()

# plt.figure(figsize=(12, 3))
# plt.plot(time[mask], Q_res_pred[mask] * 1e9, label="Pred residual (mm^3/s)")
# plt.grid(True)
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Residual [mm^3/s]")
# plt.show()

# plot the beadwidth profile

plt.figure(figsize=(12, 3))
plt.plot(time[mask], bead[mask] * 1e3, label="Beadwidth (mm)")
plt.grid(True)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Beadwidth [mm]")
plt.show()





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

# save the figure at 600 dpi

# test.savefig('/Users/james/Desktop/corner_flowmatching_NALO_test.png', bbox_inches='tight', dpi=600)


# Normalization means: {    'command':      tensor(4.8339),
#                           'bead':         tensor(2.0265),
#                           'analytical':   tensor(4.9182),
#                           'target':      tensor(-0.4706)}

# Normalization stds: {     'command':      tensor(4.0330),
#                           'bead':         tensor(0.8569),
#                           'analytical':   tensor(3.6407),
#                           'target':       tensor(1.8418)}