import numpy as np
import pandas as pd
from typing import List

from test_3D_vis import plot_bead_rect_plotly

import plotly.io as pio
pio.renderers.default = "browser"

#%%
class PathGenerator:
    def __init__(self, layer_height = 0.8, speed = 2.0, accel = 100, jerk = 0.5, dt_real = 0.01, downsample_rate = 10,
                 min_bead = 0.7, max_bead = 2.1, bead_width_override: bool = False, use_poisson_compensation: bool = False,
                 clearance_height = 5, starting_height = 30, z_offset = 0.0, flow_factor = 1.0,
                 actuator_max_delay = 0.4, actuator_max_speed = 100):
        
        # Parameters
        ##############################
        ##############################
        self.layer_height = layer_height  # print height per layer [mm]
        self.speed = speed  # toolhead speed [mm/s]
        self.accel = accel  # toolhead acceleration [mm/s^2]
        self.jerk_speed = jerk  # motion planning jerk [mm/s] allowable min speed to "jump" during accelerations
        self.dt_real = dt_real  # sampling period of robot [s]
        self.downsample_rate = downsample_rate  # downsample for constructing toolpath [1]
        self.min_bead = min_bead  # minimum bead width [mm]
        self.max_bead = max_bead  # maximum bead width [mm]
        self.bead_width_override = bead_width_override  # override the bead width with a constant value [bool]
        self.clearance_height = clearance_height  # height to raise toolhead during travel movements [mm]
        self.starting_height = starting_height  # height of the toolhead at the start of print (origin centered above the buildplate) [mm]
        self.z_offset = z_offset  # offset of the toolhead from the buildplate [mm]
        self.flow_factor = flow_factor  #
        self.actuator_max_delay = actuator_max_delay  # maximum latency between actuator commands [s]
        self.actuator_max_speed = actuator_max_speed  # maximum speed of the actuator [mm/s]
        self.use_poisson_compensation = use_poisson_compensation  # use poisson effect compensation for z height [bool]
        
        self.dt = self.dt_real / self.downsample_rate  # time step used for constructing toolpath [s]
        self.ds = self.dt * self.speed  # cartesian positional change increment [mm]
        ##############################
        ##############################
        
        self.ts = [0] # time list
        self.xs = [0] # x position list
        self.ys = [0] # y position list
        self.zs = [self.starting_height] # z position list
        self.ws = [1] # bead width list
        self.qs = [0] # flow rate list
        self.rs = [False] # scan activation list (False = not recording, True = recording active)

    def add_point(self, ts: List[float], xs: List[float], ys: List[float], zs: List[float],
                  ws: List[float], qs: List[float], rs: List[bool] = None) -> None:
        '''
        @brief mutable function that appends new points to the path
        
        Ideally, there should be no other duplicates of this function so as to keep the mutability
        of the path in one place.
        
        Parameters:
        -----------
        ts list[float]: list of time points
        xs list[float]: list of x positions
        ys list[float]: list of y positions
        zs list[float]: list of z positions
        ws list[float]: list of bead widths
        qs list[float]: list of flow rates
        rs list[bool]: list of scan activation states (False = not recording, True = recording active)
        
        @return None
        '''
        self.ts += ts
        self.xs += xs
        self.ys += ys
        self.zs += zs
        self.ws += ws
        self.qs += qs

        if rs is None:
            self.rs += [False] * len(ts)
        else:
            self.rs += rs

    def actuator_trajectory_builder(self, ts, ws, default_actuator_speed=50):

        '''
        Build an actuator velocity and position trajectory from toolpath motion data.

        Uses velocity and acceleration profiles to detect instant/trapezoidal
        increases and decreases, then pairs motion start/end indices to assign
        actuator speeds and delays. Returns a time-aligned actuator speed profile
        and adjusted width signal.

        Parameters
        ----------
        ts : array_like
            Time samples.
        ws : array_like
            Tool width or position values.
        default_actuator_speed : float, optional
            Default speed between motion events [mm/s].

        Returns
        -------
        new_ws : np.ndarray
            Adjusted width trajectory.
        vs : np.ndarray
            Actuator speed profile.
        '''

        def find_instant_decrease(acceleration, tol=1e-6):
            trajectory = np.asarray(acceleration)
            N = trajectory.shape[0]
            if N < 8:
                return []

            # Create a 2D sliding window view of shape (N - 7, 8)
            windows = np.lib.stride_tricks.sliding_window_view(trajectory, window_shape=8)

            # Boolean masks for each condition
            cond_0 = np.abs(windows[:, 0]) <= tol
            cond_1 = np.abs(windows[:, 1]) <= tol
            cond_2 = windows[:, 2] < -tol
            cond_3 = np.abs(windows[:, 3] - windows[:, 2]) <= tol
            cond_4 = windows[:, 4] > tol
            cond_5 = np.abs(windows[:, 5] - windows[:, 4]) <= tol
            cond_6 = np.abs(windows[:, 6]) <= tol
            cond_7 = np.abs(windows[:, 7]) <= tol

            # Combine all conditions
            full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7

            # Return the starting indices where the pattern matches
            return np.where(full_condition)[0]

        def find_instant_increase(acceleration, tol=1e-6):
            trajectory = np.asarray(acceleration)
            N = trajectory.shape[0]
            if N < 8:
                print("the door is stuck")
                return []

            # Create a 2D sliding window view of shape (N - 7, 8)
            windows = np.lib.stride_tricks.sliding_window_view(trajectory, window_shape=8)

            # Boolean masks for each condition
            cond_0 = np.abs(windows[:, 0]) <= tol
            cond_1 = np.abs(windows[:, 1]) <= tol
            cond_2 = windows[:, 2] > tol
            cond_3 = np.abs(windows[:, 3] - windows[:, 2]) <= tol
            cond_4 = windows[:, 4] < -tol
            cond_5 = np.abs(windows[:, 5] - windows[:, 4]) <= tol
            cond_6 = np.abs(windows[:, 6]) <= tol
            cond_7 = np.abs(windows[:, 7]) <= tol

            # Combine all conditions
            full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7

            # Return the starting indices where the pattern matches
            return np.where(full_condition)[0]

        def find_trapezoidal_increase(acceleration, velocity, tol=1e-6):
            accel = np.asarray(acceleration)
            veloc = np.asarray(velocity)
            N = accel.shape[0]
            if N < 8:
                return []

            # Create a 2D sliding window view of shape (N - 7, 8)
            accel_windows = np.lib.stride_tricks.sliding_window_view(accel, window_shape=8)
            veloc_windows = np.lib.stride_tricks.sliding_window_view(veloc, window_shape=8)

            # Boolean masks for each condition
            cond_0 = np.abs(accel_windows[:, 0]) <= tol
            cond_1 = np.abs(accel_windows[:, 1]) <= tol
            cond_2 = accel_windows[:, 2] > tol
            cond_3 = accel_windows[:, 3] > tol
            cond_4 = accel_windows[:, 4] > tol
            cond_5 = accel_windows[:, 5] > tol
            cond_6 = np.abs(accel_windows[:, 6]) <= tol
            cond_7 = np.abs(accel_windows[:, 7]) <= tol
            cond_8 = np.abs(veloc_windows[:, 7]) >= tol

            # Combine all conditions
            # full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8  # removed condition 5 because it was making the function miss some instant acceleration conditions
            full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_6 & cond_7 & cond_8

            # Return the starting indices where the pattern matches
            return np.where(full_condition)[0]

        def find_trapezoidal_decrease(acceleration, velocity, tol=1e-6):
            accel = np.asarray(acceleration)
            veloc = np.asarray(velocity)
            N = accel.shape[0]
            if N < 8:
                return []

            # Create a 2D sliding window view of shape (N - 7, 8)
            accel_windows = np.lib.stride_tricks.sliding_window_view(accel, window_shape=8)
            veloc_windows = np.lib.stride_tricks.sliding_window_view(veloc, window_shape=8)

            # Boolean masks for each condition
            cond_0 = np.abs(accel_windows[:, 0]) <= tol
            cond_1 = np.abs(accel_windows[:, 1]) <= tol
            cond_2 = accel_windows[:, 2] < -tol
            cond_3 = accel_windows[:, 3] < -tol
            cond_4 = accel_windows[:, 4] < -tol
            cond_5 = accel_windows[:, 5] < -tol
            cond_6 = np.abs(accel_windows[:, 6]) <= tol
            cond_7 = np.abs(accel_windows[:, 7]) <= tol
            cond_8 = np.abs(veloc_windows[:, 7]) >= tol

            # Combine all conditions
            # full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_5 & cond_6 & cond_7 & cond_8  # removed condition 5 because it was making the function miss some acceleration decrease conditions
            full_condition = cond_0 & cond_1 & cond_2 & cond_3 & cond_4 & cond_6 & cond_7 & cond_8

            # Return the starting indices where the pattern matches
            return np.where(full_condition)[0]

        def find_movement_starts(velocity, tol=1e-6):
            trajectory = np.asarray(velocity)
            is_zero = np.abs(trajectory) <= tol
            is_nonzero = np.abs(trajectory) > tol

            # Shift to compare v[i] == 0 and v[i+1] != 0
            transitions = is_zero[:-1] & is_nonzero[1:]

            # Return indices where transition occurs (i.e., the index of the zero value)
            return np.where(transitions)[0]

        def find_movement_ends(velocity, tol=1e-6):
            trajectory = np.asarray(velocity)
            is_nonzero = np.abs(trajectory) > tol
            is_zero = np.abs(trajectory) <= tol

            # Look for v[i] ≠ 0 and v[i+1] == 0
            transitions = is_nonzero[:-1] & is_zero[1:]

            return np.where(transitions)[0]

        def find_movement_changes(acceleration, tol=1e-6):
            trajectory = np.asarray(acceleration)
            is_nonzero = np.abs(trajectory) > tol
            is_zero = np.abs(trajectory) <= tol

            # Look for a[i] ≠ 0 and a[i+1] == 0
            starts = np.where(is_zero[:-1] & is_nonzero[1:])[0]
            ends = np.where(is_nonzero[:-1] & is_zero[1:])[0]

            if len(ends) - len(starts) == 1:
                ends = ends[1:]

            assert len(starts) == len(ends), "Mismatched number of movement changes detected."

            midpoints = np.ceil((starts+ends)/2).astype(int)


            return starts, midpoints

        def merge_and_sort_indices(*index_lists):
            # Flatten all input lists and concatenate them
            all_indices = np.concatenate([np.ravel(lst) for lst in index_lists])
            return np.sort(all_indices)

        def find_all_movements(starts, ends, changes, midpoints, exclude_near=None, exclusion_radius=8):
            starts = np.asarray(starts)
            ends = np.asarray(ends)

            if len(starts) == 0 and len(ends) == 0:
                print("No starts or ends found. Printing constant bead.")
                return [], []

            elif len(starts) == 0 or len(ends) == 0:
                raise ValueError("Start or end index list is empty.")

            if exclude_near is not None:
                exclude_near = np.asarray(exclude_near)

            pairs = []
            mask = []
            end_idx = 0

            for start in starts:
                is_near = np.any(np.abs(start - exclude_near) <= 8)
                mask.append(is_near)

                # Find the next valid end
                while end_idx < len(ends) and ends[end_idx] <= start:
                    end_idx += 1
                if end_idx >= len(ends):
                    raise ValueError(f"No corresponding end found for start index {start}.")

                pairs.append((start, ends[end_idx]))
                end_idx += 1
            # If there are still unused ends, that's also a mismatch
            if end_idx < len(ends):
                raise ValueError(f"Unpaired end indices remaining: {ends[end_idx:]}.")

            # locate all items in midpoints that are not near ant of the start or ends

            points = np.concatenate((starts, ends))
            new_points = []
            for midpoint in midpoints:
                change = changes[np.where(midpoints == midpoint)[0][0]]
                is_near = np.any(np.abs(midpoint - points) <= exclusion_radius)
                if not is_near:
                    new_points.append(change)

            for new_point in new_points:
                # find the closest start before the new_point
                valid_starts = starts[starts < new_point]
                if len(valid_starts) == 0:
                    continue
                start = valid_starts[-1]

                # find the closest end after the new_point
                valid_ends = ends[ends > new_point]
                if len(valid_ends) == 0:
                    continue
                end = valid_ends[0]

                exclude_pair = np.array([start, end])
                mask_index_to_modify = np.all(pairs == exclude_pair, axis=1)
                mask = mask * ~mask_index_to_modify

                pairs.insert(int(np.where(mask_index_to_modify)[0][0]), (start, new_point-1))
                mask = np.insert(mask, int(np.where(mask_index_to_modify)[0][0]), True)
                pairs.insert(int(np.where(mask_index_to_modify)[0][0] + 1), (new_point+1, end))
                mask = np.insert(mask, int(np.where(mask_index_to_modify)[0][0] + 1), True)

            return pairs, mask

        vel = np.gradient(ws.astype(np.float64), ts.astype(np.float64))
        acc = np.gradient(vel, ts)

        instant_decrease_indices = find_instant_decrease(acc)
        instant_increase_indices = find_instant_increase(acc)
        trapezoidal_decrease_indices = find_trapezoidal_decrease(acc, vel)
        trapezoidal_increase_indices = find_trapezoidal_increase(acc, vel)

        all_movement_starts = find_movement_starts(vel)
        all_movement_ends = find_movement_ends(vel)
        all_movement_changes, acc_midpoints = find_movement_changes(acc)

        # Merge and sort all indices
        all_indices = merge_and_sort_indices(
            instant_decrease_indices,
            instant_increase_indices,
            trapezoidal_decrease_indices,
            trapezoidal_increase_indices
        )

        # Pair the start and end indices
        try:
            all_movements, mask = find_all_movements(all_movement_starts, all_movement_ends,
                                                     all_movement_changes, acc_midpoints,
                                                     exclude_near = all_indices,
                                                     exclusion_radius = self.downsample_rate-1)
        except ValueError as e:
            print(f"Error pairing movement starts and ends: {e}. Defaulting to constant bead")
            all_movements, mask = [], []

        if len(all_movements) == 0:
            print("No movements detected. Returning constant bead width and default actuator speed.")
            new_ws = np.ones(len(ts)) * ws[0]
            vs = np.ones(len(ts)) * default_actuator_speed
            return new_ws, vs
        else:
            all_pairs = np.array(all_movements, dtype=int)
            pairs = all_pairs[np.array(mask, dtype=bool)]
            others = all_pairs[~np.array(mask, dtype=bool)]
            movement_starts = pairs[:,0]

            vs = np.ones(len(ts)) * default_actuator_speed  # default actuator speed [mm/s]
            delay_buffer = np.round(self.actuator_max_delay / self.dt_real / 2).astype(
                int)  # buffer for delay in actuator speed [points]

            for start in instant_decrease_indices:
                end = pairs[movement_starts == start + 2][0][1]
                v = np.abs(self.actuator_max_speed)
                vs[start - delay_buffer:end + delay_buffer] = v

            for start in instant_increase_indices:
                end = pairs[movement_starts == start + 2][0][1]
                v = np.abs(self.actuator_max_speed)
                vs[start - delay_buffer:end + delay_buffer] = v

            for start in trapezoidal_decrease_indices:
                end = pairs[movement_starts == start + 2][0][1]
                v = np.abs(np.round(vel[np.round((start + end) / 2).astype(int)], 3))
                vs[start - delay_buffer:end + delay_buffer] = v

            for start in trapezoidal_increase_indices:
                end = pairs[movement_starts == start + 2][0][1]
                v = np.abs(np.round(vel[np.round((start + end) / 2).astype(int)], 3))
                vs[start - delay_buffer:end + delay_buffer] = v

            new_ws = np.ones(len(ts)) * ws
            for start, end in pairs:
                new_ws[start - delay_buffer:end] = ws[end]
            # for start, end in others: # note that Falses currently mean that the pair has been rejected (and split into two corresponding pairs); in order to make this work, I'd need to filter out the rejected pairs
                # do a 0-th order discretization of the path if it's not instant or trapezoidal
                # print("Warning, this is still under construction and does not work as intended yet")
                # num_steps = int(np.floor((end - start) / delay_buffer))+2
                # points_per_step = int(np.ceil((end - start + 2 * delay_buffer) / num_steps))
                # step_levels = np.linspace(ws[start], ws[end], num_steps)
                # stair_values = np.repeat(step_levels, points_per_step)
                # new_ws[start-num_steps:start+num_steps*points_per_step] = stair_values

            return new_ws, vs

    def generate_path(self) -> pd.DataFrame:
        ts = np.array(self.ts[::self.downsample_rate])
        xs = np.array(self.xs[::self.downsample_rate])
        ys = np.array(self.ys[::self.downsample_rate])
        zs = np.array(self.zs[::self.downsample_rate]) + self.z_offset
        ws = np.array(self.ws[::self.downsample_rate])
        qs = np.array(self.qs[::self.downsample_rate]) * self.flow_factor
        rs = np.array(self.rs[::self.downsample_rate])


        if not self.bead_width_override:
            ws = np.clip(ws, self.min_bead, self.max_bead)

        if self.use_poisson_compensation:
            # zs = zs - 0.52 * (ws - self.min_bead)
            zs = zs - 0.35 * (ws - self.min_bead)
            window_size = 11
            zs = np.convolve(
                np.pad(zs, pad_width = window_size // 2, mode='edge'),
                np.ones(window_size) / window_size, mode='valid')



        ws_actual = ws
        ws_command, vs_command = self.actuator_trajectory_builder(ts, ws)

        assert np.all(np.abs(vs_command) <= self.actuator_max_speed), "Actuator speed exceeds maximum limit"

        # note on bead scanning: currently limited to a single continuous scan per path

        df = pd.DataFrame(data={
            "t": ts,
            "x": xs,
            "y": ys,
            "z": zs,
            "w": ws_command,
            "w_pred": ws_actual,
            "v_noz": vs_command,
            "q": qs,
            "r": rs
        }).round(6)
        return df

    def offset_z(self, z_offset: float) -> None:
        '''
        @brief offsets the z position

        Offsets the z position of the toolhead by a desired amount.

        Parameters:
        -----------
        z_offset [float]: desired z offset

        @return None
        '''
        self.z_offset = z_offset

    def offset_flow(self, flow_factor: float) -> None:
        """@brief adjusts flowrates by a constant factor"""
        self.flow_factor = flow_factor

    def to_gcode(self, robot_df_original, filename='test', flow_factor=1, speed_multiplier=1) -> None:
        from copy import deepcopy
        robot_df = deepcopy(robot_df_original)
        robot_df['x'] = robot_df['x'] + 100
        robot_df['y'] = robot_df['y'] + 100

        robot_df['dt'] = robot_df['t'].diff()
        robot_df['dt'] = robot_df['dt'].fillna(robot_df['dt'].mean())

        robot_df['e'] = flow_factor * robot_df['dt'] * robot_df['q'].cumsum()

        robot_df['dist'] = np.sqrt(robot_df['x'].diff() ** 2 + robot_df['y'].diff() ** 2 + robot_df['z'].diff() ** 2)
        robot_df['v_toolhead'] = speed_multiplier * 60 * robot_df['dist'] / robot_df['dt']
        robot_df['v_toolhead'] = robot_df['v_toolhead'].fillna(robot_df['v_toolhead'].mean())

        file_name = filename + '.gcode'

        startup = open('gcode_startup.txt', 'r', encoding='utf-8')
        ending = open('gcode_ending.txt', 'r', encoding='utf-8')

        gcode = open(file_name, 'w')

        for line in startup:
            gcode.write(line)
        for index, row in robot_df.iterrows():
            gcode.write('G1 X' + str(row['x']) + ' Y' + str(row['y']) + ' Z' + str(row['z']) + ' E' + str(
                row['e']) + ' F' + str(row['v_toolhead']) + '\n')
        for line in ending:
            gcode.write(line)

        startup.close()
        gcode.close()
        ending.close()

    def rectangle(self, x_traverse: float, y_traverse: float, z_height: float):
        '''
        @brief generates a rectangle path
        
        Generates rectangle path given desired x,y,z states.
        Rectangle is made in xy plane.
        
        Parameters:
        -----------
        x_traverse float: x length of rectangle
        y_traverse float: y length of rectangle
        z_height float: z height of rectangle
        
        @return None
        '''
        
        start_x = -x_traverse/2
        start_y = -y_traverse/2
        num_x_increments = round(x_traverse/self.ds)
        num_y_increments = round(y_traverse/self.ds)
        
        
        new_x = np.concatenate([
                start_x * np.ones(num_y_increments),
                np.linspace(start_x + self.ds, start_x + x_traverse, num_x_increments),
                (start_x + x_traverse) * np.ones(num_y_increments),
                np.linspace(start_x + x_traverse - self.ds, start_x, num_x_increments)
        ]).tolist()
        
        new_y = np.concatenate([
                np.linspace(start_y + self.ds, start_y + y_traverse, num_y_increments),
                (start_y + y_traverse) * np.ones(num_x_increments),
                np.linspace(start_y + y_traverse - self.ds, start_y, num_y_increments),
                start_y * np.ones(num_x_increments)
        ]).tolist()
        
        new_t = np.linspace(self.ts[-1] + self.dt, self.ts[-1] + self.dt * len(new_x), len(new_x)).tolist()
        new_z = (z_height * np.ones(len(new_x))).tolist()
        new_w = (self.wall_width * np.ones(len(new_x))).tolist()
        new_q = (self.wall_width * self.speed * self.layer_height * np.ones(len(new_x))).tolist()
        
        self.add_point(new_t, new_x, new_y, new_z, new_w, new_q)

    def print_linear(self,x_end: float, y_end: float, z_end: float, w_end: float = None ,q_end: float = None,
                     scan: bool = False) -> None:
        '''
        @brief generates a linear path from current configuration to desired end configurations
        
        This acts exactly like linspace, but conforms to the pathgen discrete time data structure.
        This enables us to generate fine discrete points conforming to the "ds" parameter of the 
        pathgen instance.
        
        Parameters:
        -----------
        x_end float: x position at the end of the linear path
        y_end float: y position at the end of the linear path
        z_end float: z position at the end of the linear path
        w_end float: bead width at the end of the linear path
        q_end float: flow rate at the end of the linear path
        
        @return None
        '''
        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        # w_start = self.ws[-1]
        # q_start = self.qs[-1]

        if w_end is None:
            w_end = self.ws[-1]

        if q_end is None:
            q_end = w_end * self.speed * self.layer_height

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start
        delta_s = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        delta_t_exact = delta_s / self.speed
        delta_t = round(delta_t_exact / self.dt) * self.dt
        
        t = np.linspace(t_start + self.dt, t_start + delta_t, round(delta_t / self.dt)).tolist()
        x = np.linspace(x_start + (len(t) and delta_x/len(t)), x_end, len(t)).tolist()
        y = np.linspace(y_start + (len(t) and delta_y/len(t)), y_end, len(t)).tolist()
        z = np.linspace(z_start + (len(t) and delta_z/len(t)), z_end, len(t)).tolist()
        w = (w_end * np.ones(len(t))).tolist()
        q = (q_end * np.ones(len(t))).tolist()
        r = ([scan] * len(t))
        
        self.add_point(t, x, y, z, w, q, r)

    def print_trapezoidal(self, x_end: float, y_end: float, z_end: float,
                          w_bead: float = None, scan: bool = False) -> None:
        '''
        @brief generates a trapezoidal path from current configuration to desired end configurations

        Generates a trapezoidal path from the current configuration to the desired end configuration.
        The trapezoidal path is generated by first accelerating to the desired speed, then moving at
        that speed for a certain distance, and finally decelerating to the end configuration.

        Parameters:
        -----------
        x_end [float]: x position at the end of the trapezoidal path
        y_end [float]: y position at the end of the trapezoidal path
        z_end [float]: z position at the end of the trapezoidal path
        w_bead [float]: bead width at the end of the trapezoidal path
        scan [bool]: whether to activate the scanner during this path segment

        @return None
        '''

        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        w_start = self.ws[-1]
        # q_start = self.qs[-1]

        layer_height = self.layer_height  # height of the layer [mm]

        if w_bead is None:
            w_bead = w_start

        max_speed = self.speed  # maximum speed of the toolhead [mm/s]
        jerk_speed = self.jerk_speed  # minimum speed to "jump" during accelerations [mm/s]
        accel = self.accel  # acceleration of the toolhead [mm/s^2]
        dt = self.dt  # time step used for constructing toolpath [s]

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start

        delta_s = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # calculate previous speed based on the delta s over the last 5 time steps
        prior_window = 5
        if len(self.ts) < prior_window:
            prior_speed = self.speed
        else:
            prior_speed = np.sqrt(
                (x_start - self.xs[-prior_window])**2 +
                (y_start - self.ys[-prior_window])**2 +
                (z_start - self.zs[-prior_window])**2) / (t_start - self.ts[-prior_window])


        delta_t_accel = (max_speed - prior_speed) / accel  # time to accelerate to speed [s]
        delta_t_decel = (max_speed - jerk_speed) / accel  # time to decelerate from speed [s]

        d_accel = accel*(delta_t_accel)*((2*t_start+delta_t_accel)/2-t_start+prior_speed/accel)
        d_decel = accel*(delta_t_decel)*((2*t_start+delta_t_decel)/2-t_start+jerk_speed/accel)

        d_steady = delta_s - (d_accel + d_decel)  # distance traveled at constant speed [mm]
        delta_t_steady = d_steady / max_speed if d_steady > 0 else 0  # time spent at constant speed [s]

        num_points_accel = round(delta_t_accel / dt)  # number of points during acceleration
        num_points_decel = round(delta_t_decel / dt)  # number of points during deceleration
        num_points_steady = round(delta_t_steady / dt)  # number of points during steady speed

        t_accel = np.linspace(
            t_start + dt,
            t_start + delta_t_accel,
            num_points_accel)
        s_accel = (
                0.5*accel*t_accel**2
                -accel*t_start*t_accel
                +prior_speed*t_accel
                +0.5*accel*t_start**2
                -prior_speed*t_start
                +0)

        t_steady = np.linspace(
            t_start + delta_t_accel + dt,
            t_start + delta_t_accel + delta_t_steady,
            num_points_steady)
        s_steady = (
                accel*delta_t_accel*t_steady
                +prior_speed*t_steady
                +0.5*accel*t_start**2
                +0.5*accel*(t_start+delta_t_accel)**2
                -accel*t_start*(t_start+delta_t_accel)
                -prior_speed*t_start
                -accel*delta_t_accel*(t_start+delta_t_accel)
                +0)

        t_decel = np.linspace(
            t_start + delta_t_accel + delta_t_steady + dt,
            t_start + delta_t_accel + delta_t_steady + delta_t_decel,
            num_points_decel)
        s_decel = (
                -0.5*accel*t_decel**2
                +accel*(t_start+delta_t_accel+delta_t_steady+delta_t_decel)*t_decel
                +jerk_speed*t_decel
                +accel*delta_t_accel*(t_start+delta_t_accel+delta_t_steady)
                +prior_speed*(t_start+delta_t_accel+delta_t_steady)
                +0.5*accel*t_start**2
                +0.5*accel*(t_start+delta_t_accel)**2
                -accel*t_start*(t_start+delta_t_accel)
                -prior_speed*t_start
                -accel*delta_t_accel*(t_start+delta_t_accel)
                +0.5*accel*(t_start+delta_t_accel+delta_t_steady)**2
                -accel*(t_start+delta_t_accel+delta_t_steady)*(t_start+delta_t_accel+delta_t_steady+delta_t_decel)
                -jerk_speed*(t_start+delta_t_accel+delta_t_steady)
                +0)

        v_accel = accel*(t_accel-t_start)+prior_speed  # velocity during acceleration phase [mm/s]
        v_steady = np.ones(num_points_steady) * accel * delta_t_accel + prior_speed  # velocity during steady phase [mm/s]
        v_decel = jerk_speed+accel*(t_start+delta_t_accel+delta_t_steady+delta_t_decel-t_decel)  # velocity during deceleration phase [mm/s]
        s = np.concatenate([s_accel, s_steady, s_decel])  # total distance traveled during trapezoidal path [mm]
        v = np.concatenate([v_accel, v_steady, v_decel])  # velocity during trapezoidal path [mm/s]

        t = np.concatenate([t_accel, t_steady, t_decel])
        x = (s*delta_x/delta_s+x_start)  # x position during trapezoidal path [mm]
        y = (s*delta_y/delta_s+y_start)  # y position during trapezoidal path [mm]
        z = (s*delta_z/delta_s+z_start)  # z position during trapezoidal path [mm]
        w = (w_bead * np.ones(len(t)))
        q = (v * w_bead * layer_height)  # velocity during trapezoidal path [mm/s]
        r = ([scan] * len(t))

        # if delta_t_steady < 0, just return the linear move function and run that instead
        # otherwise, add the points to the path
        if delta_t_steady < 0:
            print("Warning: Path is too short for trapezoidal motion planning. Using linear move instead.")
            self.print_linear(x_end=x_end, y_end=y_end, z_end=z_end, w_end=w_bead, q_end=q[-1], scan = scan)
        else:
            self.add_point(t.tolist(), x.tolist(), y.tolist(), z.tolist(), w.tolist(), q.tolist(), r)

    def travel_linear(self, x_end: float, y_end: float, z_end: float, w_end: float,
                      scan: bool = False) -> None:
        '''
        @brief generates a linear path from current configuration to desired end configurations

        This acts exactly like linspace, but conforms to the pathgen discrete time data structure.
        This enables us to generate fine discrete points conforming to the "ds" parameter of the
        pathgen instance.

        Parameters:
        -----------
        x_end float: x position at the end of the linear path
        y_end float: y position at the end of the linear path
        z_end float: z position at the end of the linear path
        w_end float: bead width at the end of the linear path
        q_end float: flow rate at the end of the linear path

        @return None
        '''
        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        # w_start = self.ws[-1]
        # q_start = self.qs[-1]


        q_end = 0.0  # flow rate is not used for travel moves, so we set it to 0.0

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start
        delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        delta_t_exact = delta_s / self.speed
        delta_t = round(delta_t_exact / self.dt) * self.dt

        t = np.linspace(t_start + self.dt, t_start + delta_t, round(delta_t / self.dt)).tolist()
        x = np.linspace(x_start + (len(t) and delta_x / len(t)), x_end, len(t)).tolist()
        y = np.linspace(y_start + (len(t) and delta_y / len(t)), y_end, len(t)).tolist()
        z = np.linspace(z_start + (len(t) and delta_z / len(t)), z_end, len(t)).tolist()
        w = (w_end * np.ones(len(t))).tolist()
        q = (q_end * np.ones(len(t))).tolist()
        r = ([scan] * len(t))

        self.add_point(t, x, y, z, w, q, r)

    def travel_trapezoidal(self, x_end: float, y_end: float, z_end: float,
                          w_bead: float = None, scan: bool = False) -> None:
        '''
        @brief generates a trapezoidal path from current configuration to desired end configurations

        Generates a trapezoidal path from the current configuration to the desired end configuration.
        The trapezoidal path is generated by first accelerating to the desired speed, then moving at
        that speed for a certain distance, and finally decelerating to the end configuration.

        Parameters:
        -----------
        x_end [float]: x position at the end of the trapezoidal path
        y_end [float]: y position at the end of the trapezoidal path
        z_end [float]: z position at the end of the trapezoidal path
        w_end [float]: bead width at the end of the trapezoidal path
        acceleration [float]: acceleration of the toolhead [mm/s^2]

        @return None
        '''

        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        w_start = self.ws[-1]
        # q_start = self.qs[-1]

        layer_height = self.layer_height  # height of the layer [mm]

        if w_bead is None:
            w_bead = w_start

        max_speed = self.speed  # maximum speed of the toolhead [mm/s]
        jerk_speed = self.jerk_speed  # minimum speed to "jump" during accelerations [mm/s]
        accel = self.accel  # acceleration of the toolhead [mm/s^2]
        dt = self.dt  # time step used for constructing toolpath [s]

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start

        delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        # calculate previous speed based on the delta s over the last 5 time steps
        prior_window = 5
        if len(self.ts) < prior_window:
            prior_speed = self.speed
        else:
            prior_speed = np.sqrt(
                (x_start - self.xs[-prior_window]) ** 2 +
                (y_start - self.ys[-prior_window]) ** 2 +
                (z_start - self.zs[-prior_window]) ** 2) / (t_start - self.ts[-prior_window])

        delta_t_accel = (max_speed - prior_speed) / accel  # time to accelerate to speed [s]
        delta_t_decel = (max_speed - jerk_speed) / accel  # time to decelerate from speed [s]

        d_accel = accel * (delta_t_accel) * ((2 * t_start + delta_t_accel) / 2 - t_start + prior_speed / accel)
        d_decel = accel * (delta_t_decel) * ((2 * t_start + delta_t_decel) / 2 - t_start + jerk_speed / accel)

        d_steady = delta_s - (d_accel + d_decel)  # distance traveled at constant speed [mm]
        delta_t_steady = d_steady / max_speed if d_steady > 0 else 0  # time spent at constant speed [s]

        num_points_accel = round(delta_t_accel / dt)  # number of points during acceleration
        num_points_decel = round(delta_t_decel / dt)  # number of points during deceleration
        num_points_steady = round(delta_t_steady / dt)  # number of points during steady speed

        t_accel = np.linspace(
            t_start + dt,
            t_start + delta_t_accel,
            num_points_accel)
        s_accel = (
                0.5 * accel * t_accel ** 2
                - accel * t_start * t_accel
                + prior_speed * t_accel
                + 0.5 * accel * t_start ** 2
                - prior_speed * t_start
                + 0)

        t_steady = np.linspace(
            t_start + delta_t_accel + dt,
            t_start + delta_t_accel + delta_t_steady,
            num_points_steady)
        s_steady = (
                accel * delta_t_accel * t_steady
                + prior_speed * t_steady
                + 0.5 * accel * t_start ** 2
                + 0.5 * accel * (t_start + delta_t_accel) ** 2
                - accel * t_start * (t_start + delta_t_accel)
                - prior_speed * t_start
                - accel * delta_t_accel * (t_start + delta_t_accel)
                + 0)

        t_decel = np.linspace(
            t_start + delta_t_accel + delta_t_steady + dt,
            t_start + delta_t_accel + delta_t_steady + delta_t_decel,
            num_points_decel)
        s_decel = (
                -0.5 * accel * t_decel ** 2
                + accel * (t_start + delta_t_accel + delta_t_steady + delta_t_decel) * t_decel
                + jerk_speed * t_decel
                + accel * delta_t_accel * (t_start + delta_t_accel + delta_t_steady)
                + prior_speed * (t_start + delta_t_accel + delta_t_steady)
                + 0.5 * accel * t_start ** 2
                + 0.5 * accel * (t_start + delta_t_accel) ** 2
                - accel * t_start * (t_start + delta_t_accel)
                - prior_speed * t_start
                - accel * delta_t_accel * (t_start + delta_t_accel)
                + 0.5 * accel * (t_start + delta_t_accel + delta_t_steady) ** 2
                - accel * (t_start + delta_t_accel + delta_t_steady) * (
                            t_start + delta_t_accel + delta_t_steady + delta_t_decel)
                - jerk_speed * (t_start + delta_t_accel + delta_t_steady)
                + 0)

        v_accel = accel * (t_accel - t_start) + prior_speed  # velocity during acceleration phase [mm/s]
        v_steady = np.ones(
            num_points_steady) * accel * delta_t_accel + prior_speed  # velocity during steady phase [mm/s]
        v_decel = jerk_speed + accel * (
                    t_start + delta_t_accel + delta_t_steady + delta_t_decel - t_decel)  # velocity during deceleration phase [mm/s]
        s = np.concatenate([s_accel, s_steady, s_decel])  # total distance traveled during trapezoidal path [mm]
        v = np.concatenate([v_accel, v_steady, v_decel])  # velocity during trapezoidal path [mm/s]

        t = np.concatenate([t_accel, t_steady, t_decel])
        x = (s * delta_x / delta_s + x_start)  # x position during trapezoidal path [mm]
        y = (s * delta_y / delta_s + y_start)  # y position during trapezoidal path [mm]
        z = (s * delta_z / delta_s + z_start)  # z position during trapezoidal path [mm]
        w = (w_bead * np.ones(len(t)))
        q = 0.0*(v * w_bead * layer_height)  # velocity during trapezoidal path [mm/s]
        r = ([scan] * len(t))

        # if delta_t_steady < 0, just return the linear move function and run that instead
        # otherwise, add the points to the path
        if delta_t_steady < 0:
            print("Warning: Path is too short for trapezoidal motion planning. Using linear move instead.")
            self.travel_linear(x_end=x_end, y_end=y_end, z_end=z_end, w_end=w_bead, q_end=q[-1], scan=scan)
        else:
            self.add_point(t.tolist(), x.tolist(), y.tolist(), z.tolist(), w.tolist(), q.tolist(), r)

    def flowmatch_trapezoidal(self, x_end: float, y_end: float, z_end: float,
                          w_bead: float = None, bead_width_override: bool = False, scan: bool = False) -> None:
        '''
        @brief generates a trapezoidal path from current configuration to desired end configurations

        Generates a trapezoidal path from the current configuration to the desired end configuration.
        The trapezoidal path is generated by first accelerating to the desired speed, then moving at
        that speed for a certain distance, and finally decelerating to the end configuration. Like print_trapezoidal,
        but adjusts the bead width time-profile to match shape of the velocity profile.

        Parameters:
        -----------
        x_end [float]: x position at the end of the trapezoidal path
        y_end [float]: y position at the end of the trapezoidal path
        z_end [float]: z position at the end of the trapezoidal path
        w_end [float]: bead width at the end of the trapezoidal path
        acceleration [float]: acceleration of the toolhead [mm/s^2]

        @return None
        '''

        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        w_start = self.ws[-1]
        # q_start = self.qs[-1]

        layer_height = self.layer_height  # height of the layer [mm]

        if w_bead is None:
            w_bead = w_start

        max_speed = self.speed  # maximum speed of the toolhead [mm/s]
        jerk_speed = self.jerk_speed  # minimum speed to "jump" during accelerations [mm/s]
        accel = self.accel  # acceleration of the toolhead [mm/s^2]
        dt = self.dt  # time step used for constructing toolpath [s]

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start

        delta_s = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # calculate previous speed based on the delta s over the last 5 time steps
        prior_window = 5
        if len(self.ts) < prior_window:
            prior_speed = self.speed
        else:
            prior_speed = np.sqrt(
                (x_start - self.xs[-prior_window])**2 +
                (y_start - self.ys[-prior_window])**2 +
                (z_start - self.zs[-prior_window])**2) / (t_start - self.ts[-prior_window])


        delta_t_accel = (max_speed - prior_speed) / accel  # time to accelerate to speed [s]
        delta_t_decel = (max_speed - jerk_speed) / accel  # time to decelerate from speed [s]

        d_accel = accel*(delta_t_accel)*((2*t_start+delta_t_accel)/2-t_start+prior_speed/accel)
        d_decel = accel*(delta_t_decel)*((2*t_start+delta_t_decel)/2-t_start+jerk_speed/accel)

        d_steady = delta_s - (d_accel + d_decel)  # distance traveled at constant speed [mm]
        delta_t_steady = d_steady / max_speed if d_steady > 0 else 0  # time spent at constant speed [s]

        num_points_accel = round(delta_t_accel / dt)  # number of points during acceleration
        num_points_decel = round(delta_t_decel / dt)  # number of points during deceleration
        num_points_steady = round(delta_t_steady / dt)  # number of points during steady speed

        t_accel = np.linspace(
            t_start + dt,
            t_start + delta_t_accel,
            num_points_accel)
        s_accel = (
                0.5*accel*t_accel**2
                -accel*t_start*t_accel
                +prior_speed*t_accel
                +0.5*accel*t_start**2
                -prior_speed*t_start
                +0)

        t_steady = np.linspace(
            t_start + delta_t_accel + dt,
            t_start + delta_t_accel + delta_t_steady,
            num_points_steady)
        s_steady = (
                accel*delta_t_accel*t_steady
                +prior_speed*t_steady
                +0.5*accel*t_start**2
                +0.5*accel*(t_start+delta_t_accel)**2
                -accel*t_start*(t_start+delta_t_accel)
                -prior_speed*t_start
                -accel*delta_t_accel*(t_start+delta_t_accel)
                +0)

        t_decel = np.linspace(
            t_start + delta_t_accel + delta_t_steady + dt,
            t_start + delta_t_accel + delta_t_steady + delta_t_decel,
            num_points_decel)
        s_decel = (
                -0.5*accel*t_decel**2
                +accel*(t_start+delta_t_accel+delta_t_steady+delta_t_decel)*t_decel
                +jerk_speed*t_decel
                +accel*delta_t_accel*(t_start+delta_t_accel+delta_t_steady)
                +prior_speed*(t_start+delta_t_accel+delta_t_steady)
                +0.5*accel*t_start**2
                +0.5*accel*(t_start+delta_t_accel)**2
                -accel*t_start*(t_start+delta_t_accel)
                -prior_speed*t_start
                -accel*delta_t_accel*(t_start+delta_t_accel)
                +0.5*accel*(t_start+delta_t_accel+delta_t_steady)**2
                -accel*(t_start+delta_t_accel+delta_t_steady)*(t_start+delta_t_accel+delta_t_steady+delta_t_decel)
                -jerk_speed*(t_start+delta_t_accel+delta_t_steady)
                +0)

        v_accel = accel*(t_accel-t_start)+prior_speed  # velocity during acceleration phase [mm/s]
        v_steady = np.ones(num_points_steady) * accel * delta_t_accel + prior_speed  # velocity during steady phase [mm/s]
        v_decel = jerk_speed+accel*(t_start+delta_t_accel+delta_t_steady+delta_t_decel-t_decel)  # velocity during deceleration phase [mm/s]
        s = np.concatenate([s_accel, s_steady, s_decel])  # total distance traveled during trapezoidal path [mm]
        v = np.concatenate([v_accel, v_steady, v_decel])  # velocity during trapezoidal path [mm/s]

        t = np.concatenate([t_accel, t_steady, t_decel])
        x = (s*delta_x/delta_s+x_start)  # x position during trapezoidal path [mm]
        y = (s*delta_y/delta_s+y_start)  # y position during trapezoidal path [mm]
        z = (s*delta_z/delta_s+z_start)  # z position during trapezoidal path [mm]

        q = (v * w_bead * layer_height)  # velocity during trapezoidal path [mm/s]
        w = (w_bead * np.clip(q/np.max(q), a_min = 0.5, a_max = None))  # bead width during trapezoidal path [mm]
        r = ([scan] * len(t))

        # ensure that w never exceeds the minimum and maximum bead width
        if not bead_width_override:
            w = np.clip(w, self.min_bead, self.max_bead)

        # if delta_t_steady < 0, just return the linear move function and run that instead
        # otherwise, add the points to the path
        if delta_t_steady < 0:
            print("Warning: Path is too short for trapezoidal motion planning. Using linear move instead.")
            self.print_linear(x_end=x_end, y_end=y_end, z_end=z_end, w_end=w_bead, q_end=q[-1], scan = scan)
        else:
            self.add_point(t.tolist(), x.tolist(), y.tolist(), z.tolist(), w.tolist(), q.tolist(), r)

    def print_beadlinear(self, x_end: float, y_end: float, z_end: float,
                       w_end: float, q_end: float = None,
                       scan: bool = False) -> None:
        '''
        @brief generates a path with constantly changing bead width while printing

        Generates a linear path from the current configuration to the desired end configuration,
        with a bead width that changes linearly from the current bead width to the desired bead width.

        Parameters:
        -----------
        x_end [float]: x position at the end of the linear path
        y_end [float]: y position at the end of the linear path
        z_end [float]: z position at the end of the linear path
        w_end [float]: bead width at the end of the linear path
        q_end [float]: flow rate override (if None, flow rate is calculated based on bead width and speed)
        scan [bool]: whether to activate the scanner during this path segment
        @return None
        '''

        t_start = self.ts[-1]
        x_start = self.xs[-1]
        y_start = self.ys[-1]
        z_start = self.zs[-1]
        w_start = self.ws[-1]

        delta_x = x_end - x_start
        delta_y = y_end - y_start
        delta_z = z_end - z_start
        delta_s = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        delta_t_exact = delta_s / self.speed
        delta_t = round(delta_t_exact / self.dt) * self.dt

        t = np.linspace(t_start + self.dt, t_start + delta_t, round(delta_t / self.dt)).tolist()
        x = np.linspace(x_start + (len(t) and delta_x / len(t)), x_end, len(t)).tolist()
        y = np.linspace(y_start + (len(t) and delta_y / len(t)), y_end, len(t)).tolist()
        z = np.linspace(z_start + (len(t) and delta_z / len(t)), z_end, len(t)).tolist()
        w = np.linspace(w_start + (len(t) and (w_end - w_start) / len(t)), w_end, len(t)).tolist()
        r = ([scan] * len(t))

        if q_end is None:
            layer_height = self.layer_height
            # todo note: currently, this only works with constant layer height
            q = (self.speed * np.array(w) * layer_height).tolist()
        else:
            q = np.linspace(0 if len(t) == 0 else self.qs[-1], q_end, len(t)).tolist()

        self.add_point(t, x, y, z, w, q, r)


    def dotted_line(self, x_end: float, y_end: float, z_end: float,
                    dot_size: float, dot_spacing: float, w_travel = 0.5) -> None:
        '''
        @brief generates a dotted line path

        Generates a dotted line path from the current configuration to the desired end configuration.

        Parameters:
        -----------
        x_end [float]: x position at the end of the dotted line
        y_end [float]: y position at the end of the dotted line
        z_end [float]: z position at the end of the dotted line
        dot_size [float]: diameter of each dot [mm]
        dot_spacing [float]: center-to-center spacing between dots [mm]

        @return None
        '''
        num_dots = np.round(
            np.sqrt((x_end - self.xs[-1])**2 + (y_end - self.ys[-1])**2 + (z_end - self.zs[-1])**2)
            / (dot_spacing))
        num_dots_real = int(np.floor(num_dots))
        x_spacing = (x_end - self.xs[-1]) / num_dots if num_dots_real > 0 else x_end - self.xs[-1]
        y_spacing = (y_end - self.ys[-1]) / num_dots if num_dots_real > 0 else y_end - self.ys[-1]
        z_spacing = (z_end - self.zs[-1]) / num_dots if num_dots_real > 0 else z_end - self.zs[-1]

        dot_volume = np.pi * (dot_size / 2)**2 * self.layer_height  # volume of each dot [mm^3]
        dot_duration = 1  # somewhat arbitrarily chosen [sec]
        dot_flow_rate = dot_volume / dot_duration  # flow rate for each dot [mm^3/s]

        for i in range(num_dots_real):
            self.print_linear(x_end =self.xs[-1] + x_spacing,
                              y_end = self.ys[-1] + y_spacing,
                              z_end = self.zs[-1] + z_spacing,
                              w_end = w_travel,
                              q_end = 0)
            self.purge(w_end = dot_size, q_end = dot_flow_rate, duration = dot_duration)

        # move to the end of the dotted line
        self.print_linear(x_end = x_end,
                          y_end = y_end,
                          z_end = z_end,
                          w_end = w_travel,
                          q_end = 0)

    def dashed_line(self, x_end: float, y_end: float, z_end: float,
                    dash_width: float, dash_length: float, dash_spacing: float, w_travel = 0.5, q_travel = 0) -> None:
        '''
        @brief generates a dashed line path

        Generates a dotted line path from the current configuration to the desired end configuration.

        Parameters:
        -----------
        x_end [float]: x position at the end of the dashed line
        y_end [float]: y position at the end of the dashed line
        z_end [float]: z position at the end of the dashed line
        dash_width [float]: width of each dash [mm]
        dash_length [float]: length of each dash [mm]
        dash_spacing [float]: center-to-center spacing between dots [mm]
        w_travel [float]: width of the VBN during travel movement [mm]

        @return None
        '''
        num_dashes = np.round(
            np.sqrt((x_end - self.xs[-1])**2 + (y_end - self.ys[-1])**2 + (z_end - self.zs[-1])**2)
            / (dash_length + dash_spacing))
        num_dashes_real = int(np.floor(num_dashes))
        x_spacing = (x_end - self.xs[-1]) / num_dashes if num_dashes_real > 0 else x_end - self.xs[-1]
        y_spacing = (y_end - self.ys[-1]) / num_dashes if num_dashes_real > 0 else y_end - self.ys[-1]
        z_spacing = (z_end - self.zs[-1]) / num_dashes if num_dashes_real > 0 else z_end - self.zs[-1]

        gap = max(0,dash_spacing - dash_length) # gap between dashes [mm]

        if num_dashes_real > 0:
            x_gap = x_spacing * (gap / dash_spacing)  # x spacing for the gap between dashes
            y_gap = y_spacing * (gap / dash_spacing)
            z_gap = z_spacing * (gap / dash_spacing)
        else:
            print("Warning: No dashes will be generated. Setting x_gap, y_gap, z_gap to 0.")
            x_gap = 0
            y_gap = 0
            z_gap = 0

        x_printlen = x_spacing * (dash_length / dash_spacing)  # x length of the printed dash
        y_printlen = y_spacing * (dash_length / dash_spacing)  # y length of the printed dash
        z_printlen = z_spacing * (dash_length / dash_spacing)  # z length of the printed dash

        dash_flow_rate = dash_width*self.layer_height*self.speed  # flow rate for each dash [mm^3/s]

        for i in range(num_dashes_real):
            self.print_linear(x_end =self.xs[-1] + x_printlen,
                              y_end = self.ys[-1] + y_printlen,
                              z_end = self.zs[-1] + z_printlen,
                              w_end = dash_width,
                              q_end = dash_flow_rate,
                              scan = True)
            self.print_linear(x_end=self.xs[-1] + x_gap,
                              y_end=self.ys[-1] + y_gap,
                              z_end=self.zs[-1] + z_gap,
                              w_end=w_travel,
                              q_end=q_travel,
                              scan = True)
        # move to the end of the dashed line
        self.print_linear(x_end = x_end,
                          y_end = y_end,
                          z_end = z_end,
                          w_end = w_travel,
                          q_end = 0)

    def purge(self, w_end: float, q_end: float, duration: float) -> None:
        '''
        @brief purges the nozzle

        Purges the nozzle by extruding a bead of width w_end and flow rate q_end for a duration of time.

        Parameters:
        -----------
        w_end [float]: bead width at the end of the purge duration [mm]
        q_end [float]: flow rate at the end of the purge duration [mm^3/s]
        duration [float]: duration of the purge [s]

        @return None
        '''
        t_start = self.ts[-1]

        t = np.linspace(t_start + self.dt, t_start + duration, round(duration / self.dt)).tolist()
        x = (self.xs[-1] * np.ones(len(t))).tolist()
        y = (self.ys[-1] * np.ones(len(t))).tolist()
        z = (self.zs[-1] * np.ones(len(t))).tolist()
        w = (w_end * np.ones(len(t))).tolist()
        q = (q_end * np.ones(len(t))).tolist()

        self.add_point(t, x, y, z, w, q)

    def dwell(self, duration: float, w_end: float) -> None:
        '''
        @brief purges the nozzle

        Purges the nozzle by extruding a bead of width w_end and flow rate q_end for a duration of time.

        Parameters:
        -----------
        w_end [float]: bead width at the end of the purge duration [mm]
        q_end [float]: flow rate at the end of the purge duration [mm^3/s]
        duration [float]: duration of the purge [s]

        @return None
        '''
        t_start = self.ts[-1]

        t = np.linspace(t_start + self.dt, t_start + duration, round(duration / self.dt)).tolist()
        x = (self.xs[-1] * np.ones(len(t))).tolist()
        y = (self.ys[-1] * np.ones(len(t))).tolist()
        z = (self.zs[-1] * np.ones(len(t))).tolist()
        w = (w_end * np.ones(len(t))).tolist()
        q = (0.0 * np.ones(len(t))).tolist()

        self.add_point(t, x, y, z, w, q)

    def rectangle_perimeters(self, z_height: float, w_end = 0.8) -> None:
        '''
        @brief generates the wall path
        
        Generates the wall path given the desired z height.
        Wall is created in the xy-plane given a desired height and 
        consists of multiple concentric rectangles that get bigger.
        
        Parameters:
        -----------
        z_height float: z height of the wall
        
        @return None
        '''
        for wall_iteration in range(self.wall_count):
            x_traverse = self.tower_xlen - self.wall_width - 2*self.wall_width*wall_iteration
            y_traverse = self.tower_ylen - self.wall_width - 2*self.wall_width*wall_iteration
            
            self.print_linear(x_end =-x_traverse / 2,
                              y_end = -y_traverse / 2,
                              z_end = z_height,
                              w_end = w_end,
                              q_end = 0)
            self.rectangle(x_traverse = x_traverse, 
                           y_traverse = y_traverse,
                           z_height = z_height)
    
    def uturn(self, start_x: float, start_y: float, uturn_spacing: float, z_height: float) -> None:
        x_traverse = uturn_spacing
        y_traverse = self.tower_ylen - 2*self.wall_count*self.wall_width-self.infill_width
        
        
        num_y_increment = round(y_traverse/self.ds)
        num_x_increment = round(x_traverse/self.ds)
        new_x = np.concatenate([
            start_x*np.ones(num_y_increment),
            np.linspace(start_x + self.ds, start_x + x_traverse, num_x_increment),
            (start_x + x_traverse)*np.ones(num_y_increment),
            np.linspace(start_x + x_traverse + self.ds, start_x + 2*x_traverse, num_x_increment)
        ]).tolist()
        new_y = np.concatenate([
            np.linspace(start_y + self.ds, start_y + y_traverse, num_y_increment),
            (start_y + y_traverse)*np.ones(num_x_increment),
            np.linspace(start_y + y_traverse + self.ds, start_y, num_y_increment),
            start_y*np.ones(num_x_increment)
        ]).tolist()
        new_z = (z_height*np.ones(len(new_x))).tolist()
        new_w = (self.infill_width*np.ones(len(new_x))).tolist()
        new_q = (self.infill_width*self.speed*self.layer_height*np.ones(len(new_x))).tolist()
        new_t = np.linspace(self.ts[-1] + self.dt, self.ts[-1] + self.dt*len(new_x), len(new_x)).tolist()
        
        self.add_point(new_t, new_x, new_y, new_z, new_w, new_q)
    
    def rectangle_infill(self, z_height: float, w_end = 0.8) -> None:
        infill_xlen = self.tower_xlen - 2*self.wall_count*self.wall_width # x-dimension of the infill portion of the tower [mm]
        num_of_uturns_exact = (infill_xlen/self.infill_width - 1)/2       # number of uturns required for the infill [1]
        num_of_uturns_whole = np.floor(num_of_uturns_exact)               # number of whole uturns that the infill will contain [1]
        infill_gap = infill_xlen - (2*num_of_uturns_whole+1)*self.infill_width # x-dimension of the gap produced by printing a whole number of uturns [mm]
        uturn_spacing_adjustment = infill_gap / (2*num_of_uturns_whole) # how much "gap" between uturns is needed for infill fill entire tower [mm]
        uturn_spacing = self.infill_width + uturn_spacing_adjustment # x-dimension of a single uturn [mm]
        
        for uturn_iteration in range(int(num_of_uturns_whole)):
            start_x = (-self.tower_xlen/2 + self.wall_width*self.wall_count + self.infill_width/2
                       + uturn_iteration*uturn_spacing*2)
            start_y = -self.tower_ylen/2 + self.wall_width*self.wall_count + self.infill_width/2
            
            self.print_linear(x_end = start_x,
                              y_end = start_y,
                              z_end = z_height,
                              w_end = w_end,
                              q_end = 0)
            
            self.uturn(start_x=start_x, 
                       start_y=start_y,
                       uturn_spacing=uturn_spacing,
                       z_height=z_height)
        
        x_new = (self.xs[-1] * np.ones(round((self.tower_ylen - 2*self.wall_count*self.wall_width - self.infill_width)
                                             /self.ds))).tolist()
        y_new = np.linspace(self.ys[-1] + self.ds, 
                            self.ys[-1] + self.tower_ylen - 2*self.wall_count*self.wall_width - self.infill_width,
                            round((self.tower_ylen - 2*self.wall_count*self.wall_width - self.infill_width)
                                  /self.ds)).tolist()
        z_new = (self.zs[-1]*np.ones(len(x_new))).tolist()
        w_new = (self.infill_width*np.ones(len(x_new))).tolist()
        q_new = (self.infill_width*self.speed*self.layer_height*np.ones(len(x_new))).tolist()
        t_new = np.linspace(self.ts[-1] + self.dt, self.ts[-1] + self.dt*len(x_new), len(x_new)).tolist()
        
        self.add_point(t_new, x_new, y_new, z_new, w_new, q_new)


def default_3DTower_gen(pathgen: PathGenerator = PathGenerator(),
                        wall_width = 0.8, infill_width = 1.6, wall_count = 2, tower_len = (40, 51, 10)) -> pd.DataFrame:

    pathgen.wall_width = wall_width  # width of wall print bead [mm]
    pathgen.infill_width = infill_width  # width of infill print bead [mm]
    pathgen.wall_count = wall_count  # number of walls per layer [1]
    pathgen.tower_xlen, pathgen.tower_ylen, pathgen.tower_zlen = tower_len  # length/width/height of tower [mm]

    num_of_layers = round(pathgen.tower_zlen / pathgen.layer_height)
    for layer_iteration in range(1, num_of_layers+1):
        pathgen.print_linear(x_end =-1 * (pathgen.tower_xlen - pathgen.wall_width) / 2,
                             y_end = -1*(pathgen.tower_ylen - pathgen.wall_width)/2,
                             z_end = pathgen.clearance_height + layer_iteration*pathgen.layer_height,
                             w_end = 0.8,
                             q_end = 0)
        pathgen.print_linear(x_end =-1 * (pathgen.tower_xlen - pathgen.wall_width) / 2,
                             y_end = -1*(pathgen.tower_ylen - pathgen.wall_width)/2,
                             z_end = pathgen.z_offset + layer_iteration*pathgen.layer_height,
                             w_end = 0.8,
                             q_end = 0)

        pathgen.rectangle_perimeters(pathgen.zs[-1], w_end = 0.8)
        pathgen.rectangle_infill(pathgen.zs[-1], w_end = 0.8)
        
    pathgen.print_linear(x_end = pathgen.xs[-1],
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.zs[-1] + pathgen.clearance_height,
                         w_end = 0.8,
                         q_end = 0)
    
    return pathgen.generate_path()

def ohbabyatriple(pathgen: PathGenerator = PathGenerator(z_offset = 1)) -> pd.DataFrame:
    ws = np.array([0.8, 1.2, 1.6])
    qs = ws * pathgen.layer_height * pathgen.speed

    # purge the nozzle
    pathgen.purge(w_end = ws[0],
                  q_end = qs[0]/2.0,
                  duration = 1.0)

    # move while changing the bead width
    for i in range(ws.shape[0]):
        pathgen.purge(w_end = ws[i],
                    q_end = qs[i],
                    duration = 20.0)
        
    return pathgen.generate_path()

def test_line_gen(pathgen: PathGenerator = PathGenerator(z_offset= 1), scan: bool = True) -> pd.DataFrame:
    # ws = np.array([0.9, 1.1, 2.0])
    # zs = np.array([0.3, 0.9, 0.8])
    # fudge_factor = np.array([1.2, 0.9, 1.0])

    ws = np.array([2.0, 1.4, 0.9])
    zs = np.array([0.8, 0.9, 0.3])
    fudge_factor = np.array([1.0, 0.9, 1.2])

    qs = ws * zs * pathgen.speed * fudge_factor

    # move down to the starting height
    pathgen.print_linear(x_end = pathgen.xs[-1] - 10.0,
                         y_end = pathgen.ys[-1] -45.0,
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0.0)

    # purge the nozzle
    pathgen.purge(w_end = ws[0],
                  q_end = qs[0]/2.0,
                  duration = 1.0)

    # move while before getting started
    pathgen.print_linear(x_end = pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = zs[0],
                         w_end = ws[0],
                         q_end = qs[0])

    # move while changing the bead width
    for i in range(ws.shape[0]):
        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1] + 30.0,
                             z_end = zs[i],
                             w_end = ws[i],
                             q_end = qs[i],
                             scan = scan)
        
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 10.0,
                          y_end = pathgen.ys[-1],
                          z_end = zs[-1],
                          w_end = ws[-1])

    # make the nozzle get out da way
    pathgen.print_linear(x_end = pathgen.xs[-1] + 20.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height + pathgen.clearance_height,
                         w_end = ws[0],
                         q_end = 0)

    return pathgen.generate_path()

def line_width_step(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:
    ws = np.array([1.6, 0.9])
    q = ws[0] * pathgen.layer_height * pathgen.speed

    # move down to the starting height
    pathgen.print_linear(x_end =pathgen.xs[-1] - 10,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)

    # purge the nozzle
    pathgen.purge(w_end = ws[0],
                  q_end = q,
                  duration = 5.0)

    # move while before getting started
    pathgen.print_linear(x_end =pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)

    # move while changing the bead width
    for i in range(ws.shape[0]):
        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1] + 40.0,
                             z_end = pathgen.layer_height,
                             w_end = ws[i],
                             q_end = q)

    # make the nozzle get out da way
    pathgen.print_linear(x_end =pathgen.xs[-1] + 10.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height + pathgen.clearance_height,
                         w_end = 0.5,
                         q_end = 0)

    return pathgen.generate_path()

def line_width_pulse(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:
    ws = np.array([1.6, 0.8, 1.6])
    q = ws[0] * pathgen.layer_height * pathgen.speed

    # move down to the starting height
    pathgen.print_linear(x_end =pathgen.xs[-1] - 10,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)

    # purge the nozzle
    pathgen.purge(w_end = ws[0],
                  q_end = q,
                  duration = 5.0)

    # move while before getting started
    pathgen.print_linear(x_end =pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)

    # move while changing the bead width
    # for i in range(ws.shape[0]):
    #     pathgen.linear_move(x_end = pathgen.xs[-1],
    #                         y_end = pathgen.ys[-1] + 40.0,
    #                         z_end = pathgen.layer_height,
    #                         w_end = ws[i],
    #                         q_end = q)

    pathgen.print_linear(x_end=pathgen.xs[-1],
                         y_end=pathgen.ys[-1] + 30.0,
                         z_end=pathgen.layer_height,
                         w_end=ws[0],
                         q_end=q)

    pathgen.print_linear(x_end=pathgen.xs[-1],
                         y_end=pathgen.ys[-1] + 2.0,
                         z_end=pathgen.layer_height,
                         w_end=ws[1],
                         q_end=q)

    pathgen.print_linear(x_end=pathgen.xs[-1],
                         y_end=pathgen.ys[-1] + 30.0,
                         z_end=pathgen.layer_height,
                         w_end=ws[-1],
                         q_end=q)

    # make the nozzle get out da way
    pathgen.print_linear(x_end =pathgen.xs[-1] + 10.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height + pathgen.clearance_height,
                         w_end = 0.5,
                         q_end = 0)

    return pathgen.generate_path()

def dotted_line_columns(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_travel = 0.5, columns_height = 5.0, column_base = 1.3, column_tip = 0.5) -> pd.DataFrame:

    no_of_layers = round(columns_height / pathgen.layer_height)
    dot_sizes = np.linspace(column_base, column_tip, no_of_layers)

    # move down to the starting height
    pathgen.print_linear(x_end =pathgen.xs[-1] - 10,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)

    # purge the nozzle
    pathgen.purge(w_end = pathgen.ws[-1],
                  q_end = 1.0,
                  duration = 5.0)

    # move while before getting started
    pathgen.print_linear(x_end =pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = w_travel,
                         q_end = 0)

    # record the current coordinates to return to later
    x_start, y_start, z_start  = pathgen.xs[-1], pathgen.ys[-1], pathgen.zs[-1]

    # call the dotted line function to create a layer of the columns

    for i in range(no_of_layers):
        pathgen.dotted_line(x_end = pathgen.xs[-1],
                            y_end = pathgen.ys[-1] + 30.0,
                            z_end = pathgen.zs[-1],
                            dot_size = dot_sizes[i],
                            dot_spacing = 5.0,
                            w_travel = w_travel)
    #
        # move up to clearance height
        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1],
                             z_end = pathgen.zs[-1] + pathgen.clearance_height,
                             w_end = w_travel,
                             q_end = 0)

        # to back to start, but overshoot in the y, and be at the next layer height
        pathgen.print_linear(x_end = x_start,
                             y_end = y_start - 10.0,
                             z_end = pathgen.zs[-1] - pathgen.clearance_height + pathgen.layer_height,
                             w_end = w_travel,
                             q_end = 0)

        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = y_start,
                             z_end = pathgen.zs[-1],
                             w_end = w_travel,
                             q_end = 0)


    # make the nozzle get out da way
    pathgen.print_linear(x_end =pathgen.xs[-1] + 10.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.zs[-1] + pathgen.clearance_height,
                         w_end = w_travel,
                         q_end = 0)

    return pathgen.generate_path()

def dashed_line_columns(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_travel = 0.5, columns_height = 5.0, column_base = 1.3, column_tip = 0.5,
                        travel_flow = 0.0, dash_length_val=1.0, dash_spacing_val=5.0) -> pd.DataFrame:

    no_of_layers = round(columns_height / pathgen.layer_height)
    dash_widths = np.linspace(column_base, column_tip, no_of_layers)
    dash_lengths = np.linspace(dash_length_val, dash_length_val, no_of_layers)
    dash_spacings = np.linspace(dash_spacing_val, dash_spacing_val, no_of_layers)

    # move down to the starting height
    pathgen.print_linear(x_end =pathgen.xs[-1] - 10,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = 1.0,
                         q_end = 0)
    # purge the nozzle
    pathgen.purge(w_end = pathgen.ws[-1],
                    q_end = 1.0,
                    duration = 5.0)
    # move while before getting started
    pathgen.print_linear(x_end =pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = w_travel,
                         q_end = 0)
    # record the current coordinates to return to later
    x_start, y_start, z_start  = pathgen.xs[-1], pathgen.ys[-1], pathgen.zs[-1]
    # call the dashed line function to create a layer of the columns
    for i in range(no_of_layers):
        pathgen.dashed_line(x_end = pathgen.xs[-1],
                            y_end = pathgen.ys[-1] + 30.0,
                            z_end = pathgen.zs[-1],
                            dash_width = dash_widths[i],
                            dash_length = dash_lengths[i],
                            dash_spacing = dash_spacings[i],
                            w_travel = w_travel,
                            q_travel = travel_flow)

        # move up to clearance height
        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1],
                             z_end = pathgen.zs[-1] + pathgen.clearance_height,
                             w_end = w_travel,
                             q_end = 0)

        # to back to start, but overshoot in the y, and be at the next layer height
        pathgen.print_linear(x_end = x_start,
                             y_end = y_start - 10.0,
                             z_end = pathgen.zs[-1] - pathgen.clearance_height + pathgen.layer_height,
                             w_end = w_travel,
                             q_end = 0)

        pathgen.print_linear(x_end = pathgen.xs[-1],
                             y_end = y_start,
                             z_end = pathgen.zs[-1],
                             w_end = w_travel,
                             q_end = 0)
    # make the nozzle get out da way
    pathgen.print_linear(x_end =pathgen.xs[-1] + 10.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.zs[-1] + pathgen.clearance_height,
                         w_end = w_travel,
                         q_end = 0)
    return pathgen.generate_path()

def bed_leveling(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                    w_end = 1.0) -> pd.DataFrame:


    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1],
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_end = w_end)

    pathgen.dwell(duration = 5.0, w_end = pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1] - 53.0,
                          z_end=pathgen.layer_height,
                          w_end=w_end)

    pathgen.dwell(duration=5.0, w_end= pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1] + 54.0,
                          y_end=pathgen.ys[-1] + 106.0,
                          z_end=pathgen.layer_height,
                          w_end=w_end)

    pathgen.dwell(duration=5.0, w_end= pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1] - 106.0,
                          y_end=pathgen.ys[-1],
                          z_end=pathgen.layer_height,
                          w_end=w_end)

    pathgen.dwell(duration=5.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1] + 54,
                          y_end=pathgen.ys[-1] - 54,
                          z_end=pathgen.clearance_height,
                          w_end=w_end)

    return pathgen.generate_path()

def corner_naive(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_print = 1.6) -> pd.DataFrame:
    '''
    @brief generates a corner flow match path

    Generates a corner flow match path from the current configuration to the desired end configuration.

    Parameters:
    -----------
    w_travel [float]: width of the VBN during travel movement [mm]
    columns_height [float]: height of the columns [mm]
    column_base [float]: base diameter of the columns [mm]
    column_tip [float]: tip diameter of the columns [mm]

    @return None
    '''

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 30,
                         y_end = pathgen.ys[-1] - 30,
                         z_end = pathgen.layer_height,
                         w_end = 1.0)

    # purge the nozzle
    pathgen.purge(w_end = pathgen.ws[-1],
                  q_end = 1.0,
                  duration = 1.0)

    # move while before getting started
    pathgen.print_trapezoidal(x_end = pathgen.xs[-1] + 60,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_bead = w_print)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 60,
                              z_end=pathgen.layer_height,
                              w_bead=w_print)

    pathgen.travel_trapezoidal(x_end=pathgen.xs[-1] - 30,
                          y_end=pathgen.ys[-1],
                          z_end=pathgen.layer_height,
                          w_bead=1.6)

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                               y_end=pathgen.ys[-1],
                               z_end=pathgen.zs[-1] + 50,
                               w_end=0.0)

    return pathgen.generate_path()

def corner_flowmatch(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_print = 1.6) -> pd.DataFrame:
    '''
    @brief generates a corner flow match path

    Generates a corner flow match path from the current configuration to the desired end configuration.

    Parameters:
    -----------
    w_travel [float]: width of the VBN during travel movement [mm]
    columns_height [float]: height of the columns [mm]
    column_base [float]: base diameter of the columns [mm]
    column_tip [float]: tip diameter of the columns [mm]

    @return None
    '''

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 30,
                         y_end = pathgen.ys[-1] - 30,
                         z_end = pathgen.layer_height,
                         w_end = 1.0)

    # purge the nozzle
    pathgen.purge(w_end = pathgen.ws[-1],
                  q_end = 1.0,
                  duration = 1.0)

    # move while before getting started
    pathgen.flowmatch_trapezoidal(x_end = pathgen.xs[-1] + 60,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height,
                         w_bead = w_print)

    pathgen.flowmatch_trapezoidal(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 60,
                              z_end=pathgen.layer_height,
                              w_bead=w_print)

    pathgen.travel_trapezoidal(x_end=pathgen.xs[-1] - 30,
                          y_end=pathgen.ys[-1],
                          z_end=pathgen.layer_height,
                          w_bead=1.6)

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                               y_end=pathgen.ys[-1],
                               z_end=pathgen.zs[-1] + 50,
                               w_end=0.0)

    return pathgen.generate_path()

def straight_line(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_print = 1.6) -> pd.DataFrame:
    '''
    @brief generates a straight line path

    Generates a straight line path from the current configuration to the desired end configuration.

    Parameters:
    -----------
    w_travel [float]: width of the VBN during travel movement [mm]
    columns_height [float]: height of the columns [mm]
    column_base [float]: base diameter of the columns [mm]
    column_tip [float]: tip diameter of the columns [mm]

    @return None
    '''

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 30,
                         y_end = pathgen.ys[-1] - 30,
                         z_end = pathgen.layer_height,
                         w_end = 1.0)

    pathgen.print_linear(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 60,
                              z_end=pathgen.layer_height,
                              w_end=w_print,
                              q_end=0.0)

    return pathgen.generate_path()

def straight_corner(pathgen: PathGenerator = PathGenerator(z_offset= 1),
                        w_print = 1.6) -> pd.DataFrame:
    '''
    @brief generates a straight line path

    Generates a straight line path from the current configuration to the desired end configuration.

    Parameters:
    -----------
    w_travel [float]: width of the VBN during travel movement [mm]
    columns_height [float]: height of the columns [mm]
    column_base [float]: base diameter of the columns [mm]
    column_tip [float]: tip diameter of the columns [mm]

    @return None
    '''

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] + 12.5,
                         y_end = pathgen.ys[-1] + 0,
                         z_end = pathgen.layer_height,
                         w_end = 1.0)

    pathgen.print_linear(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 117,
                              z_end=pathgen.layer_height,
                              w_end=w_print,
                              q_end=0.0)

    pathgen.print_linear(x_end=pathgen.xs[-1] - 12,
                         y_end=pathgen.ys[-1],
                         z_end=pathgen.layer_height,
                         w_end=w_print,
                         q_end=0.0)

    return pathgen.generate_path()


def bead_flow_constant(pathgen: PathGenerator = PathGenerator(z_offset=1), scan: bool = True) -> pd.DataFrame:

    # ws = np.array([0.70, 1.90, 1.00, 1.60, 1.30])
    # ws = np.array([0.7, 2.0, 0.8, 2.0, 0.9])
    ws = np.array([0.125, 2.0, 0.135, 2.0, 0.145])
    # ws = np.array([1.45, 1.15, 1.75, 0.85, 2.05])

    # %% Different Test Patterns

    # print("doing the 0th pattern")
    # ws = np.array([1.2, 1.2, 1.2, 1.2, 1.2])  # 0 qs = 4
    qs = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    # print("doing the 1st pattern")
    # ws = np.array([0.80, 1.80, 0.80, 1.40, 0.80]) #1 qs = 4
    # qs = np.array([4.0, 4.0, 4.0, 4.0, 4.0])

    # print("doing the 2nd pattern")
    # ws = np.array([0.75, 2.00, 0.85, 2.00, 0.95]) #2 qs = 5
    # qs = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

    # print("doing the 3rd pattern")
    # ws = np.array([1.50, 1.80, 1.10, 2.10, 0.80]) #3 qs = 6
    # qs = np.array([6.0, 6.0, 6.0, 6.0, 6.0])

    # print("doing the 4th pattern")
    # ws = np.array([2.00, 0.75, 1.70, 1.00, 1.40]) #4 qs = 7
    # qs = np.array([7.0, 7.0, 7.0, 7.0, 7.0])

    # print("doing the 5th pattern")
    # ws = np.array([1.95, 1.95, 1.35, 0.76, 0.76]) #5 qs = 7, 5, 3
    # qs = np.array([7.5, 5.5, 5.5, 3.5, 3.5])

    # print("doing the 6th pattern")
    # ws = np.array([0.80, 0.80, 1.50, 1.50, 2.10]) #6 qs = 4, 6, 8
    # qs = np.array([4.0, 4.0, 6.0, 6.0, 8.0])

    # print("doing the 7th pattern")
    # ws = np.array([1.45, 2.05, 0.78, 1.55, 1.25]) #7 qs = 4, 6, 8
    # qs = np.array([6.5, 7.7, 4.5, 6.9, 4.5])

    # print("doing the 8th pattern")
    # ws = np.array([0.79, 1.85, 1.85, 0.83, 0.83]) #8 qs = 4, 6, 8
    # qs = np.array([4.6, 7.8, 7.8, 4.7, 4.7])

    # print("doing the 9th pattern")
    # ws = np.array([1.73, 0.87, 0.87, 1.69, 1.69]) #9 qs = 4, 6, 8
    # qs = np.array([7.9, 5.3, 5.3, 7.6, 7.6])

    # print("doing the 10th pattern")
    # ws = np.array([1.88, 1.56, 1.42, 0.86, 1.97]) #10 qs = 4, 6, 8
    # qs = np.array([7.4, 6.2, 5.9, 4.3, 7.7])

    # %%

    zs = -0.52 * ws + 1.36
    # fudge_factor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    # qs = [5,10,15]
    # q = qs[0]

    # move down to the starting height
    pathgen.print_linear(x_end=pathgen.xs[-1] - 10.0,
                         y_end=pathgen.ys[-1] - 55.0,
                         z_end=pathgen.layer_height,
                         w_end=1.0,
                         q_end=0.0)

    # purge the nozzle
    pathgen.purge(w_end=ws[0],
                  q_end= 1.0,
                  duration=1.0)

    # move while before getting started
    pathgen.print_linear(x_end = pathgen.xs[-1] + 20,
                         y_end = pathgen.ys[-1],
                         z_end = zs[0],
                         w_end = ws[0],
                         q_end = ws[0]*zs[0]*pathgen.speed)

    # move while before getting started and scanning
    pathgen.print_linear(
        x_end = pathgen.xs[-1],
        y_end = pathgen.ys[-1] + 5.0,
        z_end = zs[0],
        w_end = ws[0],
        q_end = ws[0]*zs[0]*pathgen.speed,
        scan = scan)

    # move while changing the bead width
    for i in range(ws.shape[0]):
        pathgen.print_linear(x_end=pathgen.xs[-1],
                             y_end=pathgen.ys[-1] + 20.0,
                             z_end=zs[i],
                             w_end=ws[i],
                             q_end=qs[i],
                             scan=scan)

    pathgen.print_linear(
        x_end = pathgen.xs[-1],
        y_end = pathgen.ys[-1] + 5.0,
        z_end = zs[-1],
        w_end = ws[-1],
        q_end = qs[-1],
        scan = scan)

    pathgen.travel_linear(x_end = pathgen.xs[-1] - 10.0,
                          y_end = pathgen.ys[-1],
                          z_end = zs[-1],
                          w_end = 1.0)

    # make the nozzle get out da way
    pathgen.print_linear(x_end = pathgen.xs[-1] + 20.0,
                         y_end = pathgen.ys[-1],
                         z_end = pathgen.layer_height + pathgen.clearance_height,
                         w_end = 0.7,
                         q_end = 0.0)

    return pathgen.generate_path()

def sealant_static(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 20,
                          y_end = pathgen.ys[-1],
                          z_end = pathgen.layer_height,
                          w_end = 2.0)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.purge(w_end= 2.1,
                  q_end=3.5,
                  duration=4.0)

    pathgen.print_trapezoidal(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1] + 40,
                             z_end = pathgen.layer_height,
                             w_bead=2.3)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 40,
                             y_end=pathgen.ys[-1],
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1],
                             y_end=pathgen.ys[-1] - 80,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] - 40,
                             y_end=pathgen.ys[-1],
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 40,
                              z_end=pathgen.layer_height)

    pathgen.dwell(duration=5.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1] + 10,
                          z_end=pathgen.layer_height + 10,
                          w_end=2.1)

    pathgen.travel_linear(x_end=pathgen.xs[-1] + 30,
                          y_end=pathgen.ys[-1] - 15,
                          z_end=pathgen.clearance_height + 30,
                          w_end=2.1)

    return pathgen.generate_path()

def sealant_VBN(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 18.978,
                          y_end = pathgen.ys[-1] - 2.417,
                          z_end = pathgen.layer_height,
                          w_end = 1.0)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.purge(w_end= 0.9,
                  q_end=2.5,
                  duration=1.5)

    pathgen.purge(w_end=0.8,
                  q_end=2.0,
                  duration=1.5)

    pathgen.print_beadlinear(x_end = pathgen.xs[-1] - 1.022,
                             y_end = pathgen.ys[-1] + 6.600,
                             z_end = pathgen.layer_height,
                             w_end = 2.3)

    pathgen.print_trapezoidal(x_end = pathgen.xs[-1],
                              y_end = pathgen.ys[-1] + 35.817,
                              z_end = pathgen.layer_height)

    pathgen.flowmatch_trapezoidal(x_end=pathgen.xs[-1] + 40,
                                  y_end=pathgen.ys[-1],
                                  z_end=pathgen.layer_height,
                                  w_bead = 2.3)

    pathgen.flowmatch_trapezoidal(x_end=pathgen.xs[-1],
                                  y_end=pathgen.ys[-1] - 80,
                                  z_end=pathgen.layer_height,
                                  w_bead = 2.3)

    pathgen.flowmatch_trapezoidal(x_end=pathgen.xs[-1] - 40,
                                  y_end=pathgen.ys[-1],
                                  z_end=pathgen.layer_height,
                                  w_bead = 2.3)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] + 35.817,
                              z_end=pathgen.layer_height,
                              w_bead = 2.3)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] - 1.022,
                             y_end=pathgen.ys[-1] + 6.800,
                             z_end=pathgen.layer_height,
                             w_end=0.7)

    pathgen.dwell(duration=1.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1],
                          z_end=pathgen.layer_height + 1,
                          w_end=1.0)

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1] + 5,
                          z_end=pathgen.layer_height + 10,
                          w_end=1.0)

    pathgen.travel_linear(x_end=pathgen.xs[-1] + 30,
                          y_end=pathgen.ys[-1] + 15,
                          z_end=pathgen.clearance_height,
                          w_end=0.9)

    return pathgen.generate_path()

def sealant_scan(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 20,
                          y_end = pathgen.ys[-1] - 50,
                          z_end = pathgen.layer_height,
                          w_end = 2.0)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.travel_linear(x_end = pathgen.xs[-1],
                          y_end = pathgen.ys[-1] + 100,
                          z_end = pathgen.layer_height,
                          w_end=2.3,
                          scan = True)

    pathgen.dwell(duration=5.0, w_end=pathgen.ws[-1])


    pathgen.travel_linear(x_end=pathgen.xs[-1] + 20,
                          y_end=pathgen.ys[-1] - 50,
                          z_end=pathgen.clearance_height + 50,
                          w_end=0.9)

    return pathgen.generate_path()

def m_static_ideal(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 5.792,
                          y_end = pathgen.ys[-1] - 7.516,
                          z_end = pathgen.layer_height,
                          w_end = 2.0)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.purge(w_end= 2.3,
                  q_end=5.0,
                  duration=2.5)

    pathgen.print_trapezoidal(x_end = pathgen.xs[-1],
                             y_end = pathgen.ys[-1] + 15.031,
                             z_end = pathgen.layer_height,
                             w_bead=2.3)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 5.792,
                             y_end=pathgen.ys[-1] - 10.031,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 5.792,
                             y_end=pathgen.ys[-1] + 10.031,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1],
                              y_end=pathgen.ys[-1] - 15.031,
                              z_end=pathgen.layer_height)

    pathgen.dwell(duration=2.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1] + 5,
                          y_end=pathgen.ys[-1],
                          z_end=pathgen.layer_height,
                          w_end=2.0)

    pathgen.travel_linear(x_end=pathgen.xs[-1] - 30,
                          y_end=pathgen.ys[-1] - 15,
                          z_end=pathgen.clearance_height,
                          w_end=2.1)

    return pathgen.generate_path()

def m_static_naive(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 5.792,
                          y_end = pathgen.ys[-1] - 8.825,
                          z_end = pathgen.layer_height,
                          w_end = 2.1)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.purge(w_end = 2.3,
                  q_end = 5.0,
                  duration = 2.5)

    pathgen.print_linear(x_end = pathgen.xs[-1] - 0.066,
                         y_end = pathgen.ys[-1] + 11.116,
                         z_end = pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] - 0.693,
                             y_end=pathgen.ys[-1] + 7.184,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 2.649,
                             y_end=pathgen.ys[-1] - 6.648,
                             z_end=pathgen.layer_height)

    pathgen.print_linear(x_end=pathgen.xs[-1] + 2.995,
                         y_end=pathgen.ys[-1] - 5.117,
                         z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 0.906,
                             y_end=pathgen.ys[-1] - 2.835,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 0.906,
                             y_end=pathgen.ys[-1] + 2.835,
                             z_end=pathgen.layer_height)

    pathgen.print_linear(x_end=pathgen.xs[-1] + 2.995,
                              y_end=pathgen.ys[-1] + 5.117,
                              z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] + 2.649,
                             y_end=pathgen.ys[-1] + 6.648,
                             z_end=pathgen.layer_height)

    pathgen.print_trapezoidal(x_end=pathgen.xs[-1] - 0.693,
                             y_end=pathgen.ys[-1] - 7.184,
                             z_end=pathgen.layer_height)

    pathgen.print_linear(x_end=pathgen.xs[-1] - 0.066,
                         y_end=pathgen.ys[-1] - 11.116,
                         z_end=pathgen.layer_height)

    pathgen.dwell(duration=2.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1] + 5,
                          z_end=pathgen.layer_height + 5,
                          w_end=2.0)

    pathgen.travel_linear(x_end=pathgen.xs[-1] - 20,
                          y_end=pathgen.ys[-1] - 15,
                          z_end=pathgen.clearance_height + 10,
                          w_end=0.9)

    return pathgen.generate_path()

def m_VBN(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 5.792,
                          y_end = pathgen.ys[-1] - 8.825,
                          z_end = pathgen.layer_height,
                          w_end = 2.1)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.purge(w_end = 2.2,
                  q_end = 5.0,
                  duration = 2.5)

    pathgen.print_linear(x_end = pathgen.xs[-1] - 0.066,
                         y_end = pathgen.ys[-1] + 11.116,
                         z_end = pathgen.layer_height)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] - 0.693,
                             y_end=pathgen.ys[-1] + 7.184,
                             z_end=pathgen.layer_height,
                             w_end=0.8)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] + 2.649,
                             y_end=pathgen.ys[-1] - 6.648,
                             z_end=pathgen.layer_height,
                             w_end=2.2)

    pathgen.print_linear(x_end=pathgen.xs[-1] + 2.995,
                         y_end=pathgen.ys[-1] - 5.117,
                         z_end=pathgen.layer_height)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] + 0.906,
                             y_end=pathgen.ys[-1] - 2.835,
                             z_end=pathgen.layer_height,
                             w_end=0.8)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] + 0.906,
                             y_end=pathgen.ys[-1] + 2.835,
                             z_end=pathgen.layer_height,
                             w_end=2.2)

    pathgen.print_linear(x_end=pathgen.xs[-1] + 2.995,
                              y_end=pathgen.ys[-1] + 5.117,
                              z_end=pathgen.layer_height)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] + 2.649,
                             y_end=pathgen.ys[-1] + 6.648,
                             z_end=pathgen.layer_height,
                             w_end=0.8)

    pathgen.print_beadlinear(x_end=pathgen.xs[-1] - 0.693,
                             y_end=pathgen.ys[-1] - 7.184,
                             z_end=pathgen.layer_height,
                             w_end=2.2)

    pathgen.print_linear(x_end=pathgen.xs[-1] - 0.066,
                         y_end=pathgen.ys[-1] - 11.116,
                         z_end=pathgen.layer_height)

    pathgen.dwell(duration=2.0, w_end=pathgen.ws[-1])

    pathgen.travel_linear(x_end=pathgen.xs[-1],
                          y_end=pathgen.ys[-1] + 5,
                          z_end=pathgen.layer_height + 5,
                          w_end=2.0)

    pathgen.travel_linear(x_end=pathgen.xs[-1] - 20,
                          y_end=pathgen.ys[-1] - 15,
                          z_end=pathgen.clearance_height + 10,
                          w_end=0.9)

    return pathgen.generate_path()

def m_scan(pathgen: PathGenerator = PathGenerator(z_offset= 1)) -> pd.DataFrame:

    # zs = -0.52 * ws + 1.16
    # z_adjust = -0.52*(w-0.7)

    # move down to the starting height
    pathgen.travel_linear(x_end = pathgen.xs[-1] - 1.0,
                          y_end = pathgen.ys[-1] - 15,
                          z_end = pathgen.layer_height,
                          w_end = 2.0)

    pathgen.dwell(duration = 3.0, w_end = pathgen.ws[-1])

    pathgen.travel_linear(x_end = pathgen.xs[-1],
                          y_end = pathgen.ys[-1] + 30,
                          z_end = pathgen.layer_height,
                          w_end=2.3,
                          scan = True)

    pathgen.dwell(duration=5.0, w_end=pathgen.ws[-1])


    pathgen.travel_linear(x_end=pathgen.xs[-1] + 20,
                          y_end=pathgen.ys[-1] - 50,
                          z_end=pathgen.clearance_height + 50,
                          w_end=0.9)

    return pathgen.generate_path()



if __name__ == '__main__':
    pathgen = PathGenerator(speed = 3.0, accel=10, z_offset = 0.0, dt_real = 0.01, use_poisson_compensation=True, flow_factor = 0.5)
    # pathgen = PathGenerator(speed=10.0, dt_real=0.05)

    # df = straight_corner(pathgen)
    # df = default_3DTower_gen(pathgen, wall_width = 0.3, infill_width = 1.5, wall_count=4)
    # df = dashed_line_columns(pathgen, w_travel = 0.5, columns_height = 5.0, column_base = 1.3, column_tip = 0.5)
    # df = corner_flowmatch(pathgen)
    # df = test_line_gen(pathgen, scan=True)
    # df = bead_flow_constant(pathgen, scan=True)
    # df = bed_leveling(pathgen)
    df = m_VBN(pathgen)

    # pathgen.to_gcode(df,'line_test¡_coarse', flow_factor=1.05, speed_multiplier=2.0)
    # pathgen.to_gcode(df,'test', flow_factor=1.0, speed_multiplier=1.0)
    
    t = df['t'].to_numpy()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    w = df['w'].to_numpy()
    q = df['q'].to_numpy()
    r = df['r'].to_numpy()
    v_noz = df['v_noz'].to_numpy()
    w_pred = df['w_pred'].to_numpy()

    #%%
    # t = np.array(pathgen.ts)
    # x = np.array(pathgen.xs)
    # y = np.array(pathgen.ys)
    # z = np.array(pathgen.zs)
    # w = np.array(pathgen.ws)
    # q = np.array(pathgen.qs)
    # r = np.array(pathgen.rs)

    test0 = np.round(t[1:]-t[:-1],6)
    test1 = test0 == pathgen.dt_real
    if not test1.all():
        print("Warning. Hard check failed! Inconsistent time steps detected")
    else:
        print("Hard check passed: Consistent time steps is completely enforced")
    test2 = test0 - pathgen.dt_real < 0.01 * pathgen.dt_real
    test3 = np.all(test2)
    assert test3, "Soft check also failed! Time steps are not consistent enough"
    print("Soft check passed: Consistent time steps is mostly enforced")
    # calculate the total distance travelled per step
    test4 = np.zeros(np.size(t))
    for i in range(1,np.size(t)):
        test4[i] = np.sqrt((x[i]-x[i-1])**2+(y[i]-y[i-1])**2+(z[i]-z[i-1])**2)
    # test5 = test4 <= 1.02*pathgen.ds*pathgen.downsample_rate
    test5 = test4 <=0.1
    test6 = np.all(test5)
    # assert test6, "Check failed: Inconsistent speed"
    assert test6, "Safety check failed: movement faster than 100 mm/s detected"
    # print("Check passed: Constant speed is enforced")
    print("Safety check passed: No movement faster than 100 mm/s detected")
    
    import matplotlib.pyplot as plt
    plt.close('all')
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x,y,z, color = 'blue')
    ax.plot(x[:100], y[:100], z[:100], color='red')
    ax.plot(x[-1500:], y[-1500:], z[-1500:], color='green')
    ax.set_aspect('equal')
    plt.title('3D Print Path')
    plt.show()

    plt.figure()
    plt.plot(t,w_pred)
    plt.xlabel('Time [s]')
    plt.ylabel('Bead width [mm]')
    plt.title('Bead width over time')
    plt.show()

    speed = np.zeros(np.size(t))
    for i in range(1,np.size(t)):
        speed[i] = np.sqrt((x[i]-x[i-1])**2+(y[i]-y[i-1])**2+(z[i]-z[i-1])**2)/pathgen.dt_real
    plt.figure()
    plt.plot(t,speed)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [mm/s]')
    plt.title('Speed over time')
    plt.show()

    plt.figure()
    plt.plot(t, w)
    plt.xlabel('Time [s]')
    plt.ylabel('Bead width [mm]')
    plt.title('Bead command over time')
    plt.show()

    plt.figure()
    plt.plot(t,v_noz)
    plt.xlabel('Time [s]')
    plt.ylabel('Actuator speed [mm/s]')
    plt.title('Actuator speed over time')
    plt.show()

    # %%

    fig, h, A, ds = plot_bead_rect_plotly(
        df,
        width_col="w_pred",  # or "w"
        dt=pathgen.dt_real,  # you enforce constant dt_real already
        mask_mode="q>0",  # render only deposition
        ds_min=1e-4,  # tune: depends on your discretization; prevents ds~0 blowups
        h_clip=(0.0, 5.0),  # tune: keeps purge/dwell from going insane
        stride=2,  # increase (5-10) if mesh is heavy
        show_centerline=True,
        show_print_centerline=True,
    )

    #%% testing

    # test = df[df['t'] >= 7.4]
    # test['z'] = test['z'] - 2.0
    #
    # plt.close('all')
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x, y, z)
    # ax.set_aspect('equal')
    # plt.title('3D Print Path')
    # plt.show()
    # ax.plot(test['x'], test['y'], test['z'], label='Bead width')
    # test.to_csv('scan_test_2_toolpath.csv', index=False, sep=','

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    plt.figure();
    plt.plot(ds);
    plt.yscale("log");
    plt.title("ds per step (log)")
    plt.figure();
    plt.plot(q[1:] / np.maximum(ds, 1e-12));
    plt.yscale("log");
    plt.title("q/ds (log)")

    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    h_est = (q[1:] * pathgen.dt_real) / (w_pred[1:] * np.maximum(ds, 1e-12))
    spike_idx = np.where(h_est > np.percentile(h_est, 99.9))[0]
    print(spike_idx[:20])

    k = spike_idx[0]
    for ii in range(k - 3, k + 4):
        print(ii, ds[ii], q[ii + 1], w_pred[ii + 1], h_est[ii], x[ii + 1], y[ii + 1])