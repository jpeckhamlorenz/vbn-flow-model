import numpy as np
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlantConfig, AddMultibodyPlant
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.framework import DiagramBuilder
from pydrake.all import StartMeshcat, PiecewisePolynomial
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
from manipulation.scenarios import AddMultibodyTriad
# from planning import solveIK
# from diagrams import load_iiwa_setup
from inputs.vbn_pathgen import (PathGenerator,
                         test_line_gen,
                         line_width_step,
                         dotted_line_columns,
                         dashed_line_columns,
                         corner_flowmatch,
                         bed_leveling,
                         bead_flow_constant,
                         sealant_static,
                         sealant_VBN,
                         sealant_scan,
                         m_static_naive,
                         m_VBN)
import pandas as pd
from tqdm import tqdm
JOINT0 = np.array([0.0, 40 * np.pi/180, 0.0, -85 * np.pi/180, 0.0, 15 * np.pi/180, 0.0]) # NOTE: DON't change this
import torch

from sys import platform


#%% functions
def conform_pathgen(path_df: pd.DataFrame, ee_pose0: RigidTransform, rotate_z = 50.0):

    # hard code toolhead above base plate in world coords
    path_df['x'] = path_df['x']+560.0
    path_df['y'] = path_df['y']+0
    path_df['z'] = path_df['z']+266.5  # when using the buildplate, mount plate, and two rubber pads


    path_df['x'] = path_df['x']/1000
    path_df['y'] = path_df['y']/1000
    path_df['z'] = path_df['z']/1000.0
    
    rotz = RollPitchYaw(0,0, rotate_z * np.pi / 180).ToRotationMatrix()
    rotz_mat = rotz.matrix()
    
    path_xyz = path_df[["x", "y", "z"]].to_numpy().T
    path_df['x'] = rotz_mat[0,:] @ path_xyz
    path_df['y'] = rotz_mat[1,:] @ path_xyz
    path_df['z'] = rotz_mat[2,:] @ path_xyz

    # NOTE: hard coded rotation code (depends on JOINT0)
    ee_pose0.set_rotation(rotz @ RollPitchYaw(0, np.pi/2 + np.pi/6, 0).ToRotationMatrix())
    ee_pose0.set_translation(rotz.matrix() @ (ee_pose0.translation()) )
    
    return path_df, ee_pose0

def follow_pathgen(pathgen_df: pd.DataFrame, plant: MultibodyPlant, ee_pose0: RigidTransform, q0: np.ndarray, lowering_endtime = 5.0):
    print("Converting path generation to Kuka Path")
    ee_poses = [ee_pose0]
    
    lowering_steps = 10e2
    lowering_endtime = 15.0 # seconds

    # NOTE: ee_pose0 should be where you are before starting point (goto_main)
    path_lowers = np.linspace(ee_pose0.translation(), pathgen_df[["x", "y", "z"]].to_numpy()[0], round(lowering_steps))
    ts_lower = np.linspace(0, lowering_endtime, round(lowering_steps))
    print("\t generating path lowering")
    local_translation = ee_pose0.translation().copy()
    for i in tqdm(range(1, round(lowering_steps))):
        local_translation = path_lowers[i]
        ee_poses.append(RigidTransform(ee_pose0.rotation(), local_translation))
    
    print("\t generating path following")
    pathgen_positions = pathgen_df[["x", "y", "z"]].to_numpy()
    for i in tqdm(range(1,len(pathgen_df['x']))):
        ee_poses.append(RigidTransform(ee_poses[-1].rotation(), pathgen_positions[i,:]))
    
    ts = np.concatenate([ts_lower, ts_lower[-1] + pathgen_df['t'].to_numpy()[1:]])
    
    curr_q = q0.copy()
    qs = []
    print("running IK")
    for i in tqdm(range(len(ee_poses))):
        pose = ee_poses[i]
        # curr_q = solveIK(plant, pose, "iiwa_link_7", curr_q)
        qs.append(curr_q)
        
    
    qs = np.array(qs)
    q_traj = PiecewisePolynomial.FirstOrderHold(ts, qs.T)
    print("Finished converting path generation to Kuka Path")
    return q_traj, ts, qs

#%% main loop    #NOTE: modify rotation
if __name__ == '__main__':


    # meshcat = StartMeshcat()
    #
    # config = MultibodyPlantConfig()
    # config.time_step = 1e-3
    # config.penetration_allowance = 1e-9
    #
    # builder = DiagramBuilder()
    # plant, scene_graph = AddMultibodyPlant(config, builder)
    # plant: MultibodyPlant = plant  # for help w/ vscode :)
    # # load_iiwa_setup(plant, scene_graph, package_file='../package.xml')
    # plant.Finalize()
    #
    # q0 = JOINT0
    #
    # plant_context = plant.CreateDefaultContext()
    # plant.SetPositions(plant_context, q0)
    # ee_pose0 = plant.CalcRelativeTransform(plant_context, plant.world_frame(),
    #                                        plant.GetBodyByName("iiwa_link_7").body_frame())
    #
    print("Starting path generation")
    pathgen = PathGenerator(speed=1.0, accel = 50, z_offset=0.0, dt_real = 0.02, bead_width_override=True,use_poisson_compensation = True, max_bead = 2.3)
    # path_df = bead_flow_constant(pathgen)
    path_df = m_VBN(pathgen)

    # path_df = default_3DTower_gen(pathgen)
    # path_df = test_line_gen(pathgen)
    # path_df = line_width_step(pathgen)
    # path_df = dashed_line_columns(pathgen, w_travel = 0.1, columns_height = 5.0, column_base = 1.2, column_tip = 1.2, travel_flow=9.5)  # TYP: w_travel = 0.06
    # path_df, ee_pose0 = conform_pathgen(path_df, ee_pose0, rotate_z = 0.0)
    print("Finished path generation")

    if platform == 'darwin':

        from flow_predictor_lstm import flow_predictor_lstm_windowed
        from flow_predictor_analytical import flow_predictor_plots


        # ── Resolve checkpoint for windowed prediction ───────────────────
        from models.traj_WALR import get_best_run, DataModule
        from pathlib import Path as _Path

        _sweep_id = "mnywg829"
        _run_id, _run_config = get_best_run(sweep_id=_sweep_id)
        _ckpt_dir = _Path("VBN-modeling") / _run_id / "checkpoints"
        _ckpt_path = sorted(_ckpt_dir.glob("*.ckpt"))[-1]
        _dm = DataModule(_run_config, data_folderpath=_Path("dataset/recursive_samples"))
        _dm.setup("fit")

        # ── Windowed hybrid flow prediction (analytical ODE + LSTM residual)
        Q_out_naive, Q_vbn_naive, Q_res_naive = flow_predictor_lstm_windowed(
            time_np=path_df['t'].to_numpy(),
            command_np=path_df['q'].to_numpy() / 1e9,
            bead_np=path_df['w_pred'].to_numpy() / 1000,
            model_type='WALR',
            ckpt_path=_ckpt_path,
            run_config=_run_config,
            norm_stats=_dm.norm_stats,
            bead_units="m",
        )

        # ── iLQR optimization (optional: load pre-computed or run in-process)
        ILQR_RESULTS_PATH = "pathgen_ilqr_results/latest_controls.npz"  # Set to a .npz path to load pre-computed results, or None to run iLQR in-process
        RUN_ILQR = True           # Set False to skip iLQR entirely

        if ILQR_RESULTS_PATH is not None:
            from pathgen_bridge import load_ilqr_results
            print(f'Loading iLQR results from {ILQR_RESULTS_PATH}...')
            ilqr_results = load_ilqr_results(ILQR_RESULTS_PATH)
            print('\tLoaded iLQR results with keys:', ilqr_results.keys())
        elif RUN_ILQR:
            from pathgen_bridge import run_ilqr_on_pathgen
            print('Running iLQR optimization on pathgen trajectory...')
            ilqr_results = run_ilqr_on_pathgen(path_df, save_path="pathgen_ilqr_results/latest_controls.npz")
            print('\tCompleted iLQR optimization with results keys:', ilqr_results.keys())
        else:
            ilqr_results = None

        if ilqr_results is not None:

            from pathgen_bridge import smooth_segment_boundaries, smooth_Q_cmd, piecewise_linearize_w_cmd

            Q_cmd_opt_step, w_cmd_opt_step = smooth_segment_boundaries(
                t_cmd_opt = ilqr_results['t'],
                Q_cmd_opt = ilqr_results['Q_cmd_opt'],
                w_cmd_opt = ilqr_results['w_cmd_opt'],)

            Q_cmd_opt = smooth_Q_cmd(Q_cmd_opt_step, window_length=151)

            w_cmd_opt = piecewise_linearize_w_cmd(
                t = ilqr_results['t'],
                w_cmd_opt = w_cmd_opt_step,
                w_cmd_naive = ilqr_results['w_cmd_naive'],
                max_speed = 0.1,  # m/s
            )

            Q_out_opt, Q_vbn_opt, Q_res_opt = flow_predictor_lstm_windowed(
                time_np = ilqr_results['t'],
                command_np = Q_cmd_opt,
                bead_np=w_cmd_opt,
                model_type='WALR',
                ckpt_path=_ckpt_path,
                run_config=_run_config,
                norm_stats=_dm.norm_stats,
                bead_units="m",
            )

        flow_predictor_plots(
            path_df = path_df,
            ilqr_results = ilqr_results,
            Q_out_naive=Q_out_naive,
            Q_cmd_opt = Q_cmd_opt,
            w_cmd_opt = w_cmd_opt,
            Q_out_opt = Q_out_opt
        )

    
    # q_traj, ts, qs = follow_pathgen(path_df, plant, ee_pose0, q0)
    
    
    # # print(q_traj.value(0).flatten().tolist())
    # input("Press Enter to continue...")
    #
    # AddDefaultVisualization(builder, meshcat)
    # AddMultibodyTriad(plant.GetFrameByName("iiwa_link_7"), scene_graph)
    #
    # diagram = builder.Build()
    # simulator = Simulator(diagram)
    # simulator_context = simulator.get_mutable_context()
    # plant_context = plant.GetMyContextFromRoot(simulator_context)

    # meshcat.StartRecording()
    # simulator.Initialize()
    # for i in range(qs.shape[0]):
    #     simulator_context.SetTime(ts[i])
    #     plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), qs[i, :])
    #     diagram.ForcedPublish(simulator_context)
    # meshcat.StopRecording()
    # meshcat.PublishRecording()
    # input("Press to end and close plots...")
    print("\nDone.")
