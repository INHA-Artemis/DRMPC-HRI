# original file: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph

import logging
import gym
import numpy as np
import rvo2
import random
import copy
from copy import deepcopy

from numpy.linalg import norm

from environment.human_avoidance.utils.human import Human
from environment.human_avoidance.utils.robot import Robot
from environment.human_avoidance.utils.info import *
from scripts.policy.orca import ORCA
from environment.human_avoidance.utils.state import *
from environment.human_avoidance.utils.action import ActionRot, ActionXY



class CrowdSim(gym.Env):
    """
    A base environment
    treat it as an abstract class, all other environments inherit from this one
    """
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        self.config_master = None
        self.config_HA = None

        self.time_limit = None
        self.time_step = None
        self.robot = None
        # [수정] orange circle(시각장애인) 상태를 위한 별도 에이전트
        self.vip = None
        self.humans = None
        self.global_time = None
        self.global_times = []
        self.step_counter=0

        # reward function
        self.reward_strategy = None
        self.reward_params = None

        # simulation configuration
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None

        self.circle_radius = None
        self.human_num = None
        self.humans = []

        self.action_space=None
        self.observation_space=None

        # limit FOV
        self.robot_fov = None
        self.human_fov = None

        #seed
        self.thisSeed=None # the seed will be set when the env is created

        self.phase=None # set the phase to be train, val or test
        self.test_case=None # the test case ID, which will be used to calculate a seed to generate a human crossing case

        # [수정] guide pair 동적 상태(하니스 장력/오프셋)
        self.vip_offset_back = None
        self.vip_offset_right = None
        self.harness_tension = 0.0
        self.vip_anchor_prev = None
        self.vip_heading_state = None
        self.vip_gait_ratio_base = 1.0
        self.vip_gait_amp = 0.0
        self.vip_gait_freq_hz = 1.8
        self.vip_gait_phase = 0.0
        self.vip_gait_ou_state = 0.0

    def configure(self, config_master, config_HA):
        """ read the config_master to the environment variables """

        self.config_master = config_master
        self.config_HA = config_HA

        self.time_limit = config_master.config_general.env.time_limit
        self.time_step = config_master.config_general.env.time_step
        self.continuous_task = config_master.config_general.env.continuous_task
        self.frame = config_master.config_general.model.frame

        self.reward_strategy = config_HA.rewards.strategy
        self.reward_params = config_HA.rewards.params

        self.circle_radius = config_HA.sim.circle_radius
        self.circle_radius_humans = config_HA.sim.circle_radius_humans
        self.human_num_base = config_HA.sim.human_num
        self.static_human_probability, self.max_static_humans = config_HA.sim.include_static_humans['episode_probability'], config_HA.sim.include_static_humans['max_static_humans']
        self.human_num_range = config_HA.sim.human_num_range
        self.max_allowable_humans = config_HA.sim.max_allowable_humans


        self.arena_size = config_HA.sim.arena_size
        self.lookback = config_HA.sim.lookback

        self.end_goal_changing = config_HA.humans.end_goal_changing

        self.sensor_restriction_robot = config_HA.robot.sensor_restriction
        
        # set robot for this envs
        rob_RL = Robot(config_HA, 'robot')
        self.set_robot(rob_RL)
        # [수정] 기존 로봇(안내견) 외에 동반자(vip) 에이전트 생성
        self.set_vip(Robot(config_HA, 'robot'))
        if hasattr(config_HA, 'guide'):
            self.vip.radius = config_HA.guide.vip_radius
            self.vip_offset_back = config_HA.guide.offset_back_m
            self.vip_offset_right = config_HA.guide.offset_right_m
            self.vip_anchor_prev = None
            self.vip_heading_state = None

    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError

    def set_robot(self, robot):
        self.robot = robot

    def set_vip(self, vip):
        self.vip = vip

    def _body_axes(self, theta):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        e_f = np.array([cos_th, sin_th])     # forward
        e_r = np.array([sin_th, -cos_th])    # right
        return e_f, e_r

    def _vip_position_from_offsets(self, dog_pos, theta, back, right):
        e_f, e_r = self._body_axes(theta)
        return dog_pos - back * e_f + right * e_r

    def _wrap_to_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _sample_vip_gait_profile(self):
        if not hasattr(self.config_HA, 'guide'):
            return
        g = self.config_HA.guide
        if not g.vip_gait_enable:
            self.vip_gait_ratio_base = 1.0
            self.vip_gait_amp = 0.0
            self.vip_gait_freq_hz = max(g.vip_stride_freq_hz_mean, 0.1)
            self.vip_gait_phase = 0.0
            self.vip_gait_ou_state = 0.0
            return

        self.vip_gait_ratio_base = np.random.normal(g.vip_base_speed_ratio_mean, g.vip_base_speed_ratio_std)
        self.vip_gait_ratio_base = float(np.clip(self.vip_gait_ratio_base, 0.4, 1.8))

        self.vip_gait_amp = abs(np.random.normal(g.vip_stride_amp_mean, g.vip_stride_amp_std))
        self.vip_gait_amp = float(np.clip(self.vip_gait_amp, 0.0, 0.5))

        self.vip_gait_freq_hz = abs(np.random.normal(g.vip_stride_freq_hz_mean, g.vip_stride_freq_hz_std))
        self.vip_gait_freq_hz = float(np.clip(self.vip_gait_freq_hz, 0.2, 3.0))

        self.vip_gait_phase = float(np.random.uniform(0.0, 2.0 * np.pi))
        self.vip_gait_ou_state = 0.0

    def sync_vip_with_robot(self):
        # [수정] 초기화/리셋에서 vip를 dog의 우측-후방 기준값으로 정렬
        if self.vip is None:
            return
        if not hasattr(self.config_HA, 'guide') or not self.config_HA.guide.enable_pair:
            return

        self.vip_offset_back = self.config_HA.guide.offset_back_m
        self.vip_offset_right = self.config_HA.guide.offset_right_m
        dog_pos = np.array([self.robot.px, self.robot.py])
        vip_pos = self._vip_position_from_offsets(dog_pos, self.robot.theta, self.vip_offset_back, self.vip_offset_right)

        dog_goal = np.array([self.robot.gx, self.robot.gy])
        vip_goal = self._vip_position_from_offsets(dog_goal, self.robot.theta, self.vip_offset_back, self.vip_offset_right)
        self.harness_tension = 0.0

        px, py = vip_pos[0], vip_pos[1]
        gx, gy = vip_goal[0], vip_goal[1]
        self.vip.set(px, py, gx, gy, self.robot.v, self.robot.w, self.robot.theta, radius=self.config_HA.guide.vip_radius)
        self.vip.vx = 0.0
        self.vip.vy = 0.0
        self.vip_anchor_prev = vip_pos.copy()
        self.vip_heading_state = self.robot.theta
        self._sample_vip_gait_profile()

    def update_vip_with_harness_dynamics(self):
        # [수정] dog가 리드하고 vip가 우측-후방 앵커를 동적으로 추종하는 모델
        if self.vip is None:
            return
        if not hasattr(self.config_HA, 'guide') or not self.config_HA.guide.enable_pair:
            return

        g = self.config_HA.guide
        dt = self.time_step

        if self.vip_offset_back is None or self.vip_offset_right is None:
            self.vip_offset_back = g.offset_back_m
            self.vip_offset_right = g.offset_right_m
        if self.vip_heading_state is None:
            self.vip_heading_state = self.robot.theta

        dog_pos = np.array([self.robot.px, self.robot.py])
        dog_vel = np.array([self.robot.v * np.cos(self.robot.theta), self.robot.v * np.sin(self.robot.theta)])
        vip_vel = np.array([self.vip.vx, self.vip.vy])

        # [수정] heading 일치를 즉시 강제하지 않고 부드럽게 수렴
        alpha = float(np.clip(dt / max(g.heading_tau_s, 1e-6), 0.0, 1.0))
        dtheta = self._wrap_to_pi(self.robot.theta - self.vip_heading_state)
        self.vip_heading_state = self._wrap_to_pi(self.vip_heading_state + alpha * dtheta)

        e_f, e_r = self._body_axes(self.vip_heading_state)
        anchor_pos = self._vip_position_from_offsets(dog_pos, self.vip_heading_state, g.offset_back_m, g.offset_right_m)
        if self.vip_anchor_prev is None:
            self.vip_anchor_prev = anchor_pos.copy()
        anchor_vel = (anchor_pos - self.vip_anchor_prev) / dt
        self.vip_anchor_prev = anchor_pos.copy()

        vip_prev_pos = np.array([self.vip.px, self.vip.py])
        err = anchor_pos - vip_prev_pos
        v_cmd = anchor_vel + g.follow_gain * err

        # [수정] 사람별 보폭/보행 리듬 기반 속도 변동(저주파) 반영
        if g.vip_gait_enable:
            t = self.global_time
            self.vip_gait_ou_state += (
                g.vip_ou_theta * (0.0 - self.vip_gait_ou_state) * dt
                + g.vip_ou_sigma * np.sqrt(dt) * np.random.randn()
            )
            stride = self.vip_gait_amp * np.sin(2.0 * np.pi * self.vip_gait_freq_hz * t + self.vip_gait_phase)
            gait_factor = self.vip_gait_ratio_base + stride + self.vip_gait_ou_state
            gait_factor = float(np.clip(gait_factor, g.vip_gait_factor_min, g.vip_gait_factor_max))

            v_forward_cmd = float(np.dot(v_cmd, e_f))
            v_lateral_cmd = float(np.dot(v_cmd, e_r))
            v_forward_cmd = max(0.0, v_forward_cmd * gait_factor)
            v_cmd = v_forward_cmd * e_f + v_lateral_cmd * e_r

        v_cmd_norm = np.linalg.norm(v_cmd)
        if v_cmd_norm > g.vip_speed_max_mps:
            v_cmd = v_cmd / max(v_cmd_norm, 1e-6) * g.vip_speed_max_mps

        dv = v_cmd - vip_vel
        dv_norm = np.linalg.norm(dv)
        max_dv = g.vip_acc_max_mps2 * dt
        if dv_norm > max_dv:
            dv = dv / max(dv_norm, 1e-6) * max_dv
        vip_vel_new = vip_vel + dv

        # [수정] VIP는 "정지 또는 전진"만 허용 (후진 금지)
        v_forward = float(np.dot(vip_vel_new, e_f))
        v_right = float(np.dot(vip_vel_new, e_r))
        if v_forward < 0.0:
            v_forward = 0.0
        vip_vel_new = v_forward * e_f + v_right * e_r

        # [수정] 하니스는 당김만 가능: 압축(개가 사람을 미는 상황) 속도 성분 제거
        vip_rel_vec = np.array([self.vip.px, self.vip.py]) - dog_pos
        u_curr = vip_rel_vec / max(np.linalg.norm(vip_rel_vec), 1e-6)
        dL_dt_cmd = float(np.dot(vip_vel_new - dog_vel, u_curr))
        L_curr = float(np.linalg.norm(vip_rel_vec))
        if L_curr <= g.harness_length_nominal and dL_dt_cmd < 0.0:
            vip_vel_new = vip_vel_new - dL_dt_cmd * u_curr

        vip_pos_new = vip_prev_pos + vip_vel_new * dt

        rel = vip_pos_new - dog_pos
        f_comp = float(np.dot(rel, e_f))
        r_comp = float(np.dot(rel, e_r))
        back = max(g.offset_back_min_m, -f_comp)
        right = max(g.offset_right_min_m, r_comp)

        L = np.sqrt(back**2 + right**2)
        min_center_dist = self.robot.radius + self.vip.radius + g.min_pair_clearance_m
        L_min = max(g.harness_proj_length_min, min_center_dist, np.sqrt(g.offset_back_min_m**2 + g.offset_right_min_m**2))
        L_max = max(L_min + 1e-6, g.harness_proj_length_max)
        L_clamped = float(np.clip(L, L_min, L_max))
        if L > 1e-6 and abs(L_clamped - L) > 1e-9:
            scale = L_clamped / L
            back *= scale
            right *= scale

        # 장력은 preload + (신장 + 당김 속도) 기반, 0N 미만 금지
        L = np.sqrt(back**2 + right**2)
        harness_vec = -back * e_f + right * e_r
        u = harness_vec / max(np.linalg.norm(harness_vec), 1e-6)
        v_rel = float(np.dot(dog_vel - vip_vel_new, u))
        # v_rel < 0이면 dog-vip 거리가 늘어나는 방향(당김 필요), v_rel > 0은 압축 방향
        T_raw = g.min_tension_n + g.spring_k * max(L - g.harness_length_nominal, 0.0) + g.damping_c * max(-v_rel, 0.0)
        T = float(np.clip(T_raw, g.min_tension_n, g.max_tension_n))

        pull = (T / max(g.person_mass_kg, 1e-6)) * dt
        pull = float(np.clip(pull, 0.0, g.max_pull_step_m))
        back = max(g.offset_back_min_m, back - pull * (back / max(L, 1e-6)))
        right = max(g.offset_right_min_m, right - pull * (right / max(L, 1e-6)))

        # [수정] pull 적용 후에도 최소/최대 길이 제약 재강제 (겹침 방지)
        L_after_pull = np.sqrt(back**2 + right**2)
        L_after_pull_clamped = float(np.clip(L_after_pull, L_min, L_max))
        if L_after_pull > 1e-6 and abs(L_after_pull_clamped - L_after_pull) > 1e-9:
            scale = L_after_pull_clamped / L_after_pull
            back *= scale
            right *= scale

        vip_new_pos = self._vip_position_from_offsets(dog_pos, self.vip_heading_state, back, right)
        vip_goal = self._vip_position_from_offsets(np.array([self.robot.gx, self.robot.gy]), self.vip_heading_state, back, right)

        # [수정] 최종 위치에서도 후진 변위를 제거 (pivot 시 뒤로 밀림 방지)
        final_disp = vip_new_pos - vip_prev_pos
        final_forward_step = float(np.dot(final_disp, e_f))
        if final_forward_step < 0.0:
            vip_new_pos = vip_new_pos - final_forward_step * e_f

        # [수정] 최종 변위 기준으로도 사람 속도 상한 재강제
        final_disp = vip_new_pos - vip_prev_pos
        max_step = g.vip_speed_max_mps * dt
        final_disp_norm = np.linalg.norm(final_disp)
        if final_disp_norm > max_step:
            vip_new_pos = vip_prev_pos + final_disp * (max_step / max(final_disp_norm, 1e-6))

        # [수정] 최종 위치 기준에서도 압축(밀기) 성분 제거를 재강제
        final_disp = vip_new_pos - vip_prev_pos
        rel_after = vip_new_pos - dog_pos
        u_after = rel_after / max(np.linalg.norm(rel_after), 1e-6)
        dL_dt_final = float(np.dot(final_disp / dt - dog_vel, u_after))
        L_after = float(np.linalg.norm(rel_after))
        if L_after <= g.harness_length_nominal and dL_dt_final < 0.0:
            vip_new_pos = vip_new_pos - dL_dt_final * dt * u_after

        vip_vel_new = (vip_new_pos - vip_prev_pos) / dt

        self.vip_offset_back = back
        self.vip_offset_right = right
        self.harness_tension = T

        self.vip.vx = vip_vel_new[0]
        self.vip.vy = vip_vel_new[1]
        self.vip.v = np.linalg.norm(vip_vel_new)
        if self.vip.theta is None:
            self.vip.w = 0.0
        else:
            self.vip.w = self._wrap_to_pi(self.vip_heading_state - self.vip.theta) / dt

        self.vip.set(vip_new_pos[0], vip_new_pos[1], vip_goal[0], vip_goal[1], self.vip.v, self.vip.w, self.vip_heading_state, radius=g.vip_radius)
        self.vip.vx = vip_vel_new[0]
        self.vip.vy = vip_vel_new[1]

    def update_human_goal_randomly(self, human, chance):
        # chance variable is useful if we want to change the goal at random times
        if human.isObstacle:
            return
        if np.random.random() <= chance:
            humans_copy = []
            for h in self.humans:
                if h != human:
                    humans_copy.append(h)

            # Produce valid goal for human in case of circle setting
            while True:
                angle = np.random.random() * np.pi * 2
                gx_noise = (np.random.random() - 0.5) 
                gy_noise = (np.random.random() - 0.5)
                gx = self.circle_radius_humans * np.cos(angle) + gx_noise
                gy = self.circle_radius_humans * np.sin(angle) + gy_noise
                collide = False

                # [수정] 인간 목표 샘플링 충돌 검사에 dog+vip 둘 다 포함
                for agent in [self.robot, self.vip] + humans_copy:
                    min_dist = human.radius + agent.radius + 0.2
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist or norm((gx - agent.px, gy - agent.py)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            # Give human new goal
            human.gx = gx
            human.gy = gy
        return


    def generate_humans(self, path=None):
        """
        Set number of dynamic and static humans in the environment
        Return the number of humans in the environment
        """
        # dynamic human number
        self.human_num = np.random.randint(low=max(self.human_num_base - self.human_num_range, 1), high=self.human_num_base + self.human_num_range + 1)

        if self.config_HA.sim.scenario == 'circle_crossing':
            for i in range(self.human_num):
                self.humans.append(self.generate_circle_crossing_human(path))

            self.num_static_humans = 0
            if self.static_human_probability > 0:
                self.num_static_humans += self.generate_static_humans(path)

            for i in range(self.human_num + self.num_static_humans):
                self.humans[i].id = i

            return self.human_num + self.num_static_humans
        else:
            raise NotImplementedError


    def generate_static_humans(self, path=None):
        # TODO: Currently hardcoded based on paths for continuous task
        if self.continuous_task:
            human1 = Human(self.config_HA, 'humans')
            human1.set(-4, 0, 0, 0, 0, 0, 0)
            human2 = Human(self.config_HA, 'humans')
            human2.set(0, 0, 0, 0, 0, 0, 0)
            human1.isObstacle = True
            human2.isObstacle = True
            self.humans.append(human1)
            self.humans.append(human2)
            return 2

        # for episodic version
        probability_of_static_humans = np.random.random()
        if probability_of_static_humans > self.static_human_probability:
            return 0

        num_static_humans = np.random.randint(1, self.max_static_humans + 1)
        
        path_length = path.shape[1]
        for i in range(num_static_humans):
            human = Human(self.config_HA, 'humans')
            while True:
                random_idx = np.random.randint(0, path_length)
                px = self.PT_env.path[0, random_idx] + human.radius * 0.6 * np.random.uniform(-1, 1)
                py = self.PT_env.path[1, random_idx] + human.radius * 0.6 * np.random.uniform(-1, 1)

                # check minimum distance to start position
                dist_to_start = norm((px - self.PT_env.path[0, 0], py - self.PT_env.path[1, 0]))
                if dist_to_start < 0.8:
                    continue
                dist_to_end = norm((px - self.PT_env.path[0, -1], py - self.PT_env.path[1, -1]))
                if dist_to_end < 0.8:
                    continue

                collide = False

                min_buffer = 0
                if self.continuous_task:
                    min_buffer = 1 # don't want case where robot becomes sandwiched between two static humans and can't get out of soft-reset
                     

                # [수정] 정적 인간 배치 시에도 dog+vip를 하나의 확장 몸체처럼 안전거리 반영
                for i, agent in enumerate([self.robot, self.vip] + self.humans):
                    if i <= 1:
                        min_dist = human.radius + agent.radius + 0.5 + min_buffer # need this on top of checking px,py from start and end goal in case we manually start robot somewhere else
                    else:
                        min_dist = human.radius + agent.radius + 0.25 + min_buffer
                    if norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        break
                if not collide:
                    break

            human.set(px, py, -px, -py, 0, 0, 0) # the goal doesn't matter, it will be taking 0 action
            human.isObstacle = True
            self.humans.append(human)

        return num_static_humans

    # generate a human that starts on a circle, and its goal is on the opposite side of the circle
    def generate_circle_crossing_human(self, path=None):
        human = Human(self.config_HA, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()

        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            noise_range = 1.0
            px_noise = np.random.uniform(0, 1) * noise_range
            py_noise = np.random.uniform(0, 1) * noise_range
            px = self.circle_radius_humans * np.cos(angle) + px_noise
            py = self.circle_radius_humans * np.sin(angle) + py_noise
            collide = False

            # [수정] 원형 교차 시나리오 인간 생성 시 dog+vip와 최소 거리 유지
            for i, agent in enumerate([self.robot, self.vip] + self.humans):
                # keep human at least 3 meters away from robot
                if i <= 1:
                    min_dist = human.radius + agent.radius + 0.5
                else:
                    min_dist = human.radius + agent.radius + 0.2
                # if norm((px - agent.px, py - agent.py)) < min_dist or norm((px - agent.gx, py - agent.gy)) < min_dist:
                if norm((px - agent.px, py - agent.py)) < min_dist:
                    collide = True
                    break
            if not collide:
                break

        human.set(px, py, -px, -py, 0, 0, 0)

        return human

    def get_human_actions(self):
        # TODO: could add in FOV and sensor range restrictions here for humans here.
        human_actions = []  # a list of all humans' actions

        for i, human in enumerate(self.humans):
            # for static humans, just append 0 action
            if human.isObstacle:
                human_actions.append(ActionXY(0, 0))
                continue

            ob = []
            for other_human in self.humans:
                if other_human != human:
                    ob.append(other_human.get_observable_state())
                    
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            # [수정] RL 관측에는 vip를 넣지 않되, 인간 ORCA 관측에는 선택적으로 포함 가능
            #        -> 흰 원이 주황 원을 피하도록 만들기 위한 옵션
            if (
                self.vip is not None
                and self.robot.visible
                and hasattr(self.config_HA, 'guide')
                and self.config_HA.guide.humans_avoid_vip
            ):
                ob += [self.vip.get_observable_state()]
                
            human_actions.append(human.act(ob))
           
        return human_actions

    def compute_position_relative_to_robot(self, robot_x, robot_y, robot_theta, pos_x, pos_y):
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.array([robot_x, robot_y])
        p_hi_i = np.array([pos_x, pos_y])

        rel_vec = np.matmul(C_ri, p_hi_i - p_ri_i)
        return rel_vec

    def compute_position_relative_to_robot_vectorized(self, robot_x, robot_y, robot_theta, positions):
        # positions is a numpy array of Nx2
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.expand_dims(np.array([robot_x, robot_y]), 1)

        rel_vecs = np.matmul(C_ri, np.transpose(positions) - p_ri_i)
        return np.transpose(rel_vecs)
    
    def compute_pose_relative_to_robot_vectorized(self, robot_x, robot_y, robot_theta, poses):
        positions = poses[:,:2]
        thetas = np.expand_dims(poses[:,2], 1)
        cos_th = np.cos(robot_theta)
        sin_th = np.sin(robot_theta)
        C_ri = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
        p_ri_i = np.expand_dims(np.array([robot_x, robot_y]), 1)

        rel_vecs = np.matmul(C_ri, np.transpose(positions) - p_ri_i)
        rel_vecs = np.transpose(rel_vecs)
        rel_th = thetas - robot_theta          
        rel_dx = np.cos(rel_th)
        rel_dy = np.sin(rel_th)

        rel_final = np.concatenate((rel_vecs, rel_dx, rel_dy), axis=1)
        return rel_final
