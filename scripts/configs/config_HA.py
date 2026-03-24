import numpy as np
import yaml
from pathlib import Path

from .config_base import ConfigBase

class ConfigHA:
    @staticmethod
    def _load_guide_yaml():
        # [수정] 프로젝트 루트 configs/guide_dog_params.yaml에서 안내견 하이퍼파라미터 로드
        repo_root = Path(__file__).resolve().parents[2]
        yaml_path = repo_root / 'configs' / 'guide_dog_params.yaml'
        if not yaml_path.exists():
            return {}
        with open(yaml_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
        return loaded if loaded is not None else {}

    def __init__(self, config_general):
        # general configs for OpenAI gym env
        env = ConfigBase()
        env.warm_start = 5
        env.time_limit = 50
        env.time_step = config_general.env.time_step
        self.env = env

        # config_master for reward function
        rewards = ConfigBase()
        rewards.strategy = ['disturbance', 'collision', 'actuation_termination', 'safety_human_raise']

        if config_general.env.continuous_task is False:
            assert config_general.use_time is True
            rewards.strategy.append('timeout')

            if config_general.use_PEB:
                rewards.timeout_penalty = 0
            else:
                rewards.timeout_penalty = -5

        rewards.params = {
            'actuation_termination': {'min_vel': 0.025, 'penalty': -20},
            'safety_human_raise': {'safety_dist': 0.22, 'penalty': -15},
            'collision': {'penalty': -15},
            'disturbance': {
                'radius': 1.5,
                'factor_v': 0.8 * 7,
                'factor_th': 0.5 * 7
            }
        }
        self.rewards = rewards

        # config_master for simulation
        sim = ConfigBase()
        sim.scenario = 'circle_crossing'
        sim.circle_radius = 4
        sim.circle_radius_humans = 5
        sim.arena_size = max(sim.circle_radius, sim.circle_radius_humans) + 0.25
        sim.human_num = 6
        sim.include_static_humans = {'episode_probability': 0, 'max_static_humans': 0} # ex. for continuous task (human positions preset for now): {'episode_probability': 0, 'max_static_humans': 2 } 
        sim.human_num_range = 0


        sim.max_allowable_humans = sim.human_num + sim.human_num_range
        if sim.include_static_humans['episode_probability'] > 0:
            sim.max_allowable_humans += sim.include_static_humans['max_static_humans']

        sim.lookback = config_general.env.lookback
        sim.warm_start = True if sim.lookback > 0 else False
        self.sim = sim

        # human config_master
        humans = ConfigBase()
        humans.visible = True

        # orca or social_force for now
        humans.policy = "orca"
        humans.kinematics = "holonomic"
        if humans.policy == "orca":
            assert humans.kinematics == "holonomic"
        humans.radius = 0.3
        humans.v_max = 1.0

        humans.end_goal_changing = True
        humans.randomize_attributes = False # I haven't verified this in yet.
        humans.sensor_FOV = None # not supported right now
        humans.sensor_range = None # not supported right now


        
        self.humans = humans

        # robot config_master
        robot = ConfigBase()
        # whether robot is visible to humans (whether humans respond to the robot's motion)
        robot.visible = True # JH: personally I think this should never be False since invisible testbed defeats RL's purpose of learning how the robot's action will influence other humans
        robot.radius = 0.3
        robot.v_max = config_general.robot.v_max
        robot.v_min = config_general.robot.v_min
        robot.w_max = config_general.robot.w_max
        robot.w_min = config_general.robot.w_min
        robot.sensor_restriction = False # TODO: if True, verify correctness
        robot.sensor_FOV = 2 * np.pi
        robot.sensor_range = 4
        
        robot.kinematics = "unicycle" # options ['holonomic', 'unicycle', 'unicycle_with_lag']
        robot.unicycle_with_lag_params = {'max_lin_acc': 1.5, 'max_ang_acc': 2.0}

        if robot.sensor_restriction and 'disturbance' in rewards.strategy:
            assert robot.sensor_range > rewards.params['disturbance']['radius'], "Sensor range must be greater than disturbance penalty calculation radius."

        self.robot = robot

        guide_yaml = self._load_guide_yaml()

        # [수정] 안내견-시각장애인 페어 설정
        guide = ConfigBase()
        guide.enable_pair = bool(guide_yaml.get('enable_pair', True))
        guide.vip_radius = float(guide_yaml.get('vip_radius', 0.3))

        # [수정] VIP 기본 위치(개 body frame 기준): 오른쪽 30cm, 뒤 10cm
        guide.offset_right_m = float(guide_yaml.get('offset_right_m', 0.30))
        guide.offset_back_m = float(guide_yaml.get('offset_back_m', 0.10))
        guide.offset_right_min_m = float(guide_yaml.get('offset_right_min_m', 0.05))
        guide.offset_back_min_m = float(guide_yaml.get('offset_back_min_m', 0.02))

        # [수정] 3D 강체 하니스(길이 1m) 각도 범위(0도=지면과 평행)를 2D 투영 길이 범위로 변환
        guide.rod_length_m = float(guide_yaml.get('rod_length_m', 1.0))
        guide.rod_angle_deg_min = float(guide_yaml.get('rod_angle_deg_min', 30.0))
        guide.rod_angle_deg_max = float(guide_yaml.get('rod_angle_deg_max', 90.0))
        guide.harness_proj_length_max = guide.rod_length_m * np.cos(np.deg2rad(guide.rod_angle_deg_min))
        guide.harness_proj_length_min = guide.rod_length_m * np.cos(np.deg2rad(guide.rod_angle_deg_max))
        guide.harness_length_nominal = float(np.sqrt(guide.offset_right_m**2 + guide.offset_back_m**2))

        # [수정] 동적 모델 하이퍼파라미터(미정값은 안전 기본값으로 시작)
        guide.person_mass_kg = float(guide_yaml.get('person_mass_kg', 65.0))
        guide.spring_k = float(guide_yaml.get('spring_k', 120.0))
        guide.damping_c = float(guide_yaml.get('damping_c', 45.0))
        guide.max_tension_n = float(guide_yaml.get('max_tension_n', 350.0))
        guide.min_tension_n = float(guide_yaml.get('min_tension_n', 15.0))
        guide.min_pair_clearance_m = float(guide_yaml.get('min_pair_clearance_m', 0.02))
        guide.max_pull_step_m = float(guide_yaml.get('max_pull_step_m', 0.03))
        guide.follow_gain = float(guide_yaml.get('follow_gain', 2.8))
        guide.vip_speed_max_mps = float(guide_yaml.get('vip_speed_max_mps', 1.5))
        guide.vip_acc_max_mps2 = float(guide_yaml.get('vip_acc_max_mps2', 2.5))
        guide.heading_tau_s = float(guide_yaml.get('heading_tau_s', 0.8))
        guide.dog_no_pivot_min_v_mps = float(guide_yaml.get('dog_no_pivot_min_v_mps', 0.15))
        guide.vip_gait_enable = bool(guide_yaml.get('vip_gait_enable', True))
        guide.vip_base_speed_ratio_mean = float(guide_yaml.get('vip_base_speed_ratio_mean', 1.0))
        guide.vip_base_speed_ratio_std = float(guide_yaml.get('vip_base_speed_ratio_std', 0.1))
        guide.vip_stride_amp_mean = float(guide_yaml.get('vip_stride_amp_mean', 0.1))
        guide.vip_stride_amp_std = float(guide_yaml.get('vip_stride_amp_std', 0.04))
        guide.vip_stride_freq_hz_mean = float(guide_yaml.get('vip_stride_freq_hz_mean', 1.8))
        guide.vip_stride_freq_hz_std = float(guide_yaml.get('vip_stride_freq_hz_std', 0.2))
        guide.vip_ou_theta = float(guide_yaml.get('vip_ou_theta', 1.2))
        guide.vip_ou_sigma = float(guide_yaml.get('vip_ou_sigma', 0.1))
        guide.vip_gait_factor_min = float(guide_yaml.get('vip_gait_factor_min', 0.6))
        guide.vip_gait_factor_max = float(guide_yaml.get('vip_gait_factor_max', 1.4))

        # [수정] 비-하이퍼파라미터 요구사항(항상 True)
        guide.humans_avoid_vip = True
        guide.render_harness = True
        guide.render_tension_text = True
        self.guide = guide

        # config_master for ORCA
        orca = ConfigBase()
        orca.neighbor_dist = 10
        orca.safety_space = 0.175
        orca.time_horizon = 5
        orca.time_horizon_obst = 5
        self.orca = orca
