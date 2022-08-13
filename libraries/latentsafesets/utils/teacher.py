from abc import ABC
from turtle import distance

from latentsafesets.envs import simple_point_bot as spb
from latentsafesets.envs import simple_velocity_bot as svb
from latentsafesets.envs import bottleneck_nav as bnn

import numpy as np
import math


class AbstractTeacher(ABC):

    def __init__(self, env, noisy=False, on_policy=True, horizon=None):
        self.env = env
        self.noisy = noisy
        self.on_policy = on_policy

        self.ac_high = env.action_space.high
        self.ac_low = env.action_space.low
        self.noise_std = (self.ac_high - self.ac_low) / 20
        self.random_start = False
        if horizon is None:
            self.horizon = env.horizon
        else:
            self.horizon = horizon

    def generate_demonstrations(self, num_demos, store_noisy=True, noise_param=None):
        demonstrations = []
        for i in range(num_demos):
            demo = self.generate_trajectory(noise_param, store_noisy=store_noisy)
            reward = sum([frame['reward'] for frame in demo])
            print('Trajectory %d, Reward %d' % (i, reward))
            demonstrations.append(demo)
        return demonstrations

    def generate_trajectory(self, noise_param=None, store_noisy=True):
        """
        The teacher initially tries to go northeast before going to the origin
        """
        self.reset()
        transitions = []
        obs = self.env.reset(random_start=self.random_start)
        # state = np.zeros((0, 0))
        state = None
        done = False
        for i in range(self.horizon):
            if state is None:
                action = self.env.action_space.sample().astype(np.float64)
            else:
                action = self._expert_control(state, i).astype(np.float64)
            if self.noisy:
                action_input = np.random.normal(action, self.noise_std)
                action_input = np.clip(action_input, self.ac_low, self.ac_high)
            else:
                action_input = action

            if store_noisy:
                action = action_input
            action_input = action_input.astype(np.float32)
            next_obs, reward, done, info = self.env.step(action_input)
            transition = {'obs': obs, 'action': tuple(action), 'reward': float(reward),
                          'next_obs': next_obs, 'done': int(done),
                          'constraint': int(info['constraint']), 'safe_set': 0,
                          'on_policy': int(self.on_policy)}
            # print({k: v.dtype for k, v in transition.items() if 'obs' in k})
            transitions.append(transition)
            state = info['next_state']
            obs = next_obs

            if done:
                break

        transitions[-1]['done'] = 1

        rtg = 0
        ss = 0
        for frame in reversed(transitions):
            if frame['reward'] >= 0:
                ss = 1

            frame['safe_set'] = ss
            frame['rtg'] = rtg

            rtg = rtg + frame['reward']

        # assert done, "Did not reach the goal set on task completion."
        # V = self.env.values()
        # for i, t in enumerate(transitions):
        #     t['values'] = V[i]
        return transitions

    def _expert_control(self, state, i):
        raise NotImplementedError("Override in subclass")

    def reset(self):
        pass


class SimplePointBotTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, noisy)
        self.goal = (150, 75)

    def _expert_control(self, s, t):
        if t < 20:
            goal = np.array((30, 15))
        elif t < 60:
            goal = np.array((150, 15))
        else:
            goal = self.goal

        act = np.subtract(goal, s)
        act = np.clip(act, -3, 3)
        return act


class ConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE
        self.goal = (88, 75)
        self.random_start = True

    def _expert_control(self, state, i):
        if i < 15:
            return self.d
        else:
            to_obstactle = np.subtract(self.goal, state)
            to_obstacle_normalized = to_obstactle / np.linalg.norm(to_obstactle)
            to_obstactle_scaled = to_obstacle_normalized * spb.MAX_FORCE / 2
            return to_obstactle_scaled

    def reset(self):
        self.d = (np.random.random(2) * 2 - 1) * spb.MAX_FORCE


class SimpleVelocityBotTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, noisy)
        self.goal = (150, 75)
        self.start_pos = (30, 75)
        self.goal_state = 0
        self.goal_set = {0: np.array(self.start_pos), 1: np.array((30, 15)), 2: np.array((150, 15)), 3: np.array(self.goal)}
        self.control_state = 0
        self.filet_radius = 10.0
        self.xv_error_sum, self.yv_error_sum = 0.0, 0.0
        self.last_xv_error, self.last_yv_error = 0.0, 0.0

    @staticmethod
    def _bearing(a, b):
        point_diff = np.subtract(a, b)
        return math.atan2(point_diff[0], point_diff[1])

    @staticmethod
    def _distance(a, b):
        point_diff = np.subtract(a, b)
        return np.linalg.norm(point_diff)

    @staticmethod
    def _check_angle(bearing):
        while bearing < 0.0:
            bearing += 2.0 * np.pi
        while bearing > 2.0 * np.pi:
            bearing -= 2.0 * np.pi
        return bearing

    @staticmethod
    def _check_angle_negative_range(bearing):
        while bearing > np.pi:
            bearing -= 2.0 * np.pi
        while bearing < -np.pi:
            bearing += 2.0 * np.pi
        return bearing    
    
    @staticmethod
    def _unit_dir_vector(start_point, end_point):
        direction_vector = np.subtract(start_point, end_point)
        try:
            unit_vector_n = direction_vector[0] / math.sqrt(math.pow(direction_vector[0], 2)
                            + math.pow(direction_vector[1], 2))
        except ZeroDivisionError:
            unit_vector_n = 0
        try:
            unit_vector_e = direction_vector[1] / math.sqrt(math.pow(direction_vector[0], 2)
                            + math.pow(direction_vector[1], 2))
        except ZeroDivisionError:
            unit_vector_e = 0
        unit_vector = (unit_vector_n, unit_vector_e)
        return unit_vector

    def _expert_control(self, s, t):
        if t <= 2:
            self.goal_state = 0
            self.control_state = 0
        if self.goal_state < 2:
            final_state = False
            point_a = self.goal_set[self.goal_state]
            point_b = self.goal_set[self.goal_state + 1]
            point_c = self.goal_set[self.goal_state + 2]        
            track_bearing_in = self._check_angle(self._bearing(point_a, point_b))
            track_bearing_out = self._check_angle(self._bearing(point_b, point_c))
            filet_angle = self._check_angle(track_bearing_out - track_bearing_in)
            if self.control_state == 0:
                q = self._unit_dir_vector(point_a, point_b)
                w = point_b
                try:
                    z_point = (w[0] - ((self.filet_radius / math.tan(filet_angle / 2)) * q[0]),
                               w[1] - ((self.filet_radius / math.tan(filet_angle / 2)) * q[1]))
                    h_point = np.subtract(z_point, s[:2])
                    h_val = (h_point[0] * q[0]) + (h_point[1] * q[1])
                    if h_val > 0:
                        # Entered h-plane transition to curved segment
                        self.control_state = 1
                    
                    objective_bearing = self._bearing(s[:2], point_b)
                    objective_distance = self._distance(s[:2], point_b)
                    track_distance = self._distance(point_a, point_b)
                    off_track_angle = self._check_angle_negative_range(objective_bearing - track_bearing_in)
                    heading = (0.5 * track_distance/objective_distance * off_track_angle) + track_bearing_in
                except ZeroDivisionError:
                    print(f'Straight lines between: {point_a}, {point_b}, {point_c}')

            if self.control_state == 1:
                q0 = self._unit_dir_vector(point_a, point_b)
                q1 = self._unit_dir_vector(point_b, point_c)
                q_grad = self._unit_dir_vector(q0, q1)
                w = point_b
                center_point = (w[0] - ((self.filet_radius / math.sin(filet_angle / 2 * (math.pi / 180.0))) * q_grad[0]),
                                w[1] - ((self.filet_radius / math.sin(filet_angle / 2 * (math.pi / 180.0))) * q_grad[1]))
                z_point = (w[0] + ((self.filet_radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q1[0]),
                           w[1] + ((self.filet_radius / math.tan(filet_angle / 2 * (math.pi / 180.0))) * q1[1]))
                turning_direction = math.copysign(1, (q0[0] * q1[1]) - (q0[1] * q1[0]))
                h_point = np.subtract(z_point, s[:2])
                h_val = (h_point[0] * q1[0]) + (h_point[1] * q1[1])
                if h_val > 0:
                    self.control_state = 0
                    self.goal_state += 1
                distance_from_center = math.sqrt(math.pow(s[0] - center_point[0], 2) +
                                                math.pow(s[1] - center_point[1], 2))
                circ_x = s[1] - center_point[1]
                circ_y = s[0] - center_point[0]
                circle_angle = math.atan2(circ_x, circ_y)
                if circle_angle < 0:
                        circle_angle = circle_angle + (2 * math.pi)
                tangent_track = circle_angle + (turning_direction * (math.pi / 2))
                if tangent_track < 0:
                    tangent_track = tangent_track + (2 * math.pi)
                if tangent_track > 2 * math.pi:
                    tangent_track = tangent_track - (2 * math.pi)
                error = (distance_from_center - self.filet_radius) / self.filet_radius
                k_orbit = 4.0
                heading = tangent_track - (math.atan(k_orbit * error))
        else:
            final_state = True
            heading = self._bearing(s[:2], self.goal_set[3])
            dist = self._distance(s[:2], self.goal_set[3])
            if dist < 20.0:
                final_state = True
                speed_dem = (0.075 * dist ** 2)
            elif dist < 10.0:
                final_state = True
                speed_dem = 0.0
            else:
                speed_dem = 3.0
        
        speed_dem = 4.0
        xv_dem = -speed_dem * math.sin(heading)
        yv_dem = -speed_dem * math.cos(heading)
        xv_error = s[2] - xv_dem
        yv_error = s[3] - yv_dem

        xv_der_error = self.last_xv_error - xv_error
        yv_der_error = self.last_yv_error - yv_error
        self.last_xv_error = xv_error
        self.last_yv_error = yv_error

        px, py = 5.0, 5.0
        pix, piy = 0.05, 0.05
        pdx, pdy = 0.0, 0.0
        if final_state:
            px, py = 5.0, 5.0
            pix, piy = 0.5, 0.5
            pdx, pdy = 4.0, 4.0
        x_accel_dem = - (px * xv_error) - (pix * self.xv_error_sum) + (pdx * xv_der_error)
        y_accel_dem = - (py * yv_error) - (piy * self.yv_error_sum) + (pdy * yv_der_error)

        self.xv_error_sum += xv_error
        self.yv_error_sum += yv_error
        

        act = np.array([x_accel_dem, y_accel_dem])
        act = np.clip(act, a_min=-svb.MAX_ACCEL, a_max=svb.MAX_ACCEL)
        return act


class SimpleVelocityBotConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * svb.MAX_ACCEL
        self.goal = (88, 75)
        self.random_start = True
    
    def _expert_control(self, state, t):
        if t < 15:
            return self.d
        else:
            to_obstacle = np.subtract(state[:2], self.goal)
            bearing = math.atan2(to_obstacle[0], to_obstacle[1])
            # print(f'bearing to target: {bearing * (180.0 / np.pi)}')
            speed_dem = 10.0
            xv_dem = -speed_dem * math.sin(bearing)
            yv_dem = -speed_dem * math.cos(bearing)
            xv_error = state[2] - xv_dem
            yv_error = state[3] - yv_dem
            px, py = 5.0, 5.0
            x_accel_dem = - px * xv_error
            y_accel_dem = - py * yv_error
            act = np.array([x_accel_dem, y_accel_dem])
            act = np.clip(act, a_min=-svb.MAX_ACCEL, a_max=svb.MAX_ACCEL)
            return act
        
    def reset(self):
        self.d = (np.random.random(2) * 2 -1) * svb.MAX_ACCEL

class BottleNeckTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, noisy)
        self.goal = env.goal

    def _expert_control(self, state, t):
        goal = self.goal
        action = np.subtract(goal, state)
        action = np.clip(action, -bnn.MAX_FORCE, bnn.MAX_FORCE)
        if np.NaN in action:
            print(action)
        return action


class BottleNeckConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super().__init__(env, noisy, on_policy=False)
        self.d = (np.random.random(2) * 2 - 1) * bnn.MAX_FORCE
        self.random_start = True
        self.x_act = 0.0
        self.y_act = 0.0

    # TODO: Check that the direction on this changes at each iter by monitoring i value
    def _expert_control(self, state, t):
        if t < 15:
            # Wonder around for a bit
            return self.d
        if t==16:
            direction = np.random.uniform(-math.pi, math.pi)
            self.x_act = np.sqrt(bnn.MAX_FORCE**2/(math.tan(direction)**2 + 1))
            self.y_act = np.tan(direction) * self.x_act
            if math.pi/2 < direction or direction < -math.pi/2:
                self.x_act *= -1
            print(f'x_act is: {self.x_act}, y_act is: {self.y_act}')
            print(f'!!! Direction is: {direction * 180/math.pi} !!!')
        action = np.array([self.x_act, self.y_act])
        if np.NaN in action:
            print(action)
        return action


class ReacherTeacher(AbstractTeacher):
    def __init__(self, env, noisy=True):
        super().__init__(env, noisy=noisy, horizon=100)

    def _expert_control(self, state, i):
        if i < 40:
            goal = np.array((np.pi, 0))
        else:
            goal = np.array((np.pi * .75, 0))

        angle = state[:2]
        act = goal - angle
        act = np.clip(act, -1, 1)
        return act


class ReacherConstraintTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(ReacherConstraintTeacher, self).__init__(env, noisy, on_policy=False)
        self.direction = 1
        self.random_start = True

    def _expert_control(self, state, i):
        angle = state[:2]
        goal1 = np.array((np.pi * .53, 0.7 * np.pi))
        goal2 = np.array((np.pi, -0.7 * np.pi))
        goal = min(goal1, goal2, key=lambda x: np.linalg.norm(angle - x))
        act = goal - angle
        # act = np.random.normal((self.direction, 0), 1)
        act = np.clip(act, -1, 1)
        return act

    def reset(self):
        self.direction = self.direction * -1


class PushTeacher(AbstractTeacher):

    def __init__(self, env, noisy):
        super(PushTeacher, self).__init__(env, False)
        self.demonstrations = []
        self.default_noise = 0.2
        self.block_id = 0
        self.horizon = 150

    def _expert_control(self, state, i):
        action, block_done = self.env.expert_action(block=self.block_id, noise_std=0.004)
        if block_done:
            self.block_id += 1
            self.block_id = min(self.block_id, 2)

        return action

    def reset(self):
        self.block_id = 0


class StrangeTeacher(AbstractTeacher):
    def __init__(self, env, noisy=False):
        super(StrangeTeacher, self).__init__(env, noisy, on_policy=False)
        self.d_act = env.action_space.shape
        self.high = env.action_space.high
        self.low = env.action_space.low
        self.std = (self.high - self.low) / 10
        self.last_action = env.action_space.sample()
        self.random_start = True
        self.horizon = 20

    def _expert_control(self, state, i):
        action = np.random.normal(self.last_action, self.std)
        action = np.clip(action, self.low, self.high)
        self.last_action = action
        return action

    def reset(self):
        self.last_action = self.env.action_space.sample()


class OutburstPushTeacher(AbstractTeacher):
    def __init__(self, env, noisy):
        super(OutburstPushTeacher, self).__init__(env, False, False)
        # self.block_id = 0
        self.horizon = 150
        self.outburst = False

    def _expert_control(self, state, i):
        if np.random.random() > .8:
            self.outburst = True

        if np.random.random() > .9:
            self.outburst = False

        if self.outburst:
            return self.env.action_space.sample().astype(np.float64)

        return np.array((0, -0.02))

    def reset(self):
        self.block_id = 0
        self.outburst = False



