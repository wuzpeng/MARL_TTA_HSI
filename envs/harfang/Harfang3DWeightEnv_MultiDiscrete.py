"""
harfang 3D_weight 6v6空战环境
动作设置为底层舵偏
多目标策略，奖励函数加权求和
"""
import logging
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from random import random, choice
import dogfight_client as df
from Constants import *
import math
from math import pi, atan2
from gymnasium.utils import seeding
import time
import gym
from EnvSocket import EnvSocketServer


class MultiHarfangEnv:
    def __init__(self, port):
        df.connect("127.0.0.1", port)  # TODO:Change IP and PORT values
        time.sleep(2)
        # True: 设置无渲染模式  False: 默认有渲染模式
        df.set_renderless_mode(False)
        df.set_client_update_mode(True)

        self.np_random = None
        self.num_episode = 0  # 总局数
        self.num_win = 0      # 赢的局数
        self.num_destroy_enemy = 0  # 击毁敌机数
        self.sum_destroy_enemy = 0  # 击毁敌机总和
        self.current_step = 0  # 当前步数
        self.sum_steps = 0  # 步数总和
        self.attack_angle = pi/30   # 角度：6度  弧度：pi/30
        self.attack_dist = 1100       # 攻击距离
        self.base_speed = 300
        self.fixed_height = 4000
        self.num_UAVs = 6
        self.num_allyUAVs = 6
        self.num_oppoUAVs = 6

        # 敌方观测噪声设置：5% 随机观测噪声
        # 注意：只用于策略观测，不修改真实环境状态
        self.enable_enemy_obs_noise = False
        self.enemy_obs_noise_ratio = 0  # 噪声大小

        # 敌方位置观测时延设置：只用于策略观测，不修改真实环境状态
        self.enable_enemy_obs_delay = False
        self.enemy_obs_delay_sec = 0  # 可设置为 0.1, 0.3, 0.5, 1.0
        self.enemy_obs_delay_speed = 120.0  # 敌方速度，单位 m/s

        # TTA输出扰动设置：随机选择一个attention weight元素增加0.5
        # 注意：只扰动TTA输出/目标优先级，不修改真实环境状态
        self.enable_attention_weight_perturb = False
        self.attention_weight_perturb_value = 0.25

        self.ally_state_n = []       # 存储己方飞机状态信息
        self.enemy_state_n = []      # 存储敌方飞机状态信息
        self.old_ally_state_n = []   # 存储己方飞机上一步的状态信息
        self.old_enemy_state_n = []  # 存储敌方飞机上一步的状态信息
        self.Plane_ID_ally = []      # ["ally_1", "ally_2" ,"ally_3", ... , "ally_15 "]
        self.Plane_ID_oppo = []      # ["ennemy_1", "ennemy_2" ,"ennemy_3", ... , "ennemy_15 "]

        for i in range(1, self.num_UAVs + 1):
            self.Plane_ID_oppo.append("ennemy_" + str(i))
            self.Plane_ID_ally.append("ally_" + str(i))

        self.dones = [[False] for _ in range(self.num_allyUAVs)]
        self.attention_weight = np.array([[1 / self.num_oppoUAVs] for _ in range(self.num_oppoUAVs)])

        # 己方动作
        self.ally_action_spaces = gym.spaces.MultiDiscrete([11, 11, 11])  # [滚转、俯仰、方向舵]
        # 敌方动作
        # self.oppo_action_spaces = gym.spaces.MultiDiscrete([11, 11, 11])

        # 个体观测，共享观测
        ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n = self.reset()
        self.ally_obs_spaces = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(ally_obs_n.shape[1],), dtype=np.float64)
        self.ally_share_obs_spaces = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(ally_share_obs.shape[1],),
                                                    dtype=np.float64)
        # self.ennemy_obs_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float64)

        self.human_preference = [0.0 for _ in range(self.num_oppoUAVs)]
        self.UIServer = EnvSocketServer(host='127.0.0.1', port=9999)
        self.UIServer.env = self

    def _get_plane_state(self):
        self.ally_state_n = []  # 每一步清空己方状态信息
        self.enemy_state_n = []  # 每一步清空敌方状态信息
        for plane_id in range(self.num_UAVs):
            ally_state = df.get_plane_state(self.Plane_ID_ally[plane_id])  # TCP通信从环境中获取己方状态
            enemy_state = df.get_plane_state(self.Plane_ID_oppo[plane_id])  # TCP通信从环境中获取敌方状态

            self.ally_state_n.append(ally_state)
            self.enemy_state_n.append(enemy_state)

    def reset(self):  # reset simulation beginning of episode
        self.current_step = 0
        self.num_destroy_enemy = 0
        self._get_plane_state()
        self.old_ally_state_n = self.ally_state_n
        self.old_enemy_state_n = self.enemy_state_n
        self.attention_weight = np.array([[1 / self.num_oppoUAVs] for _ in range(self.num_oppoUAVs)])
        self.human_preference = [0.0 for _ in range(self.num_oppoUAVs)]

        # 将权重映射为整数优先级，数字越大优先级越大
        int_attention_weight = self.int_attention_weight(self.human_preference)

        self._reset_machine()
        ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n = self._get_attention_observation(int_attention_weight)

        return ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n

    def close(self):
        pass

    def _reset_machine(self):
        x = [2000, 4000, 7000]  # 可供选取的初始化位置
        random_x_ally = choice(x)  # 随机初始化位置
        for ally_id in range(self.num_allyUAVs):
            init_x_ally = 7000 + 300 * ally_id  # 计算每一个己方飞机初始化位置
            # 重置己方飞机状态与位姿
            df.rearm_machine(self.Plane_ID_ally[ally_id])
            df.reset_machine(self.Plane_ID_ally[ally_id])
            df.retract_gear(self.Plane_ID_ally[ally_id])  # 飞机起落架
            df.set_plane_linear_speed(self.Plane_ID_ally[ally_id], self.base_speed)  # 设置起始速度
            df.reset_machine_matrix(self.Plane_ID_ally[ally_id], init_x_ally, self.fixed_height, 1000, 0, 0, 0)

        random_x_enemy = choice(x)
        for enemy_id in range(self.num_oppoUAVs):
            init_x_enemy = random_x_enemy + 500 * enemy_id  # 计算每一个敌方飞机初始化位置
            # 重置敌方飞机状态与位姿
            df.rearm_machine(self.Plane_ID_oppo[enemy_id])
            df.reset_machine(self.Plane_ID_oppo[enemy_id])
            df.retract_gear(self.Plane_ID_oppo[enemy_id])  # 飞机起落架
            df.set_plane_linear_speed(self.Plane_ID_oppo[enemy_id], self.base_speed - 180)  # 设置起始速度
            # df.reset_machine_matrix(self.Plane_ID_oppo[enemy_id], init_x_enemy, self.fixed_height, 9000, 0, pi, 0)

        # 2-2-2 分布
        df.reset_machine_matrix("ennemy_1", 1750, self.fixed_height, 9000, 0, pi, 0)
        df.reset_machine_matrix("ennemy_2", 2250, self.fixed_height, 9000, 0, pi, 0)
        df.reset_machine_matrix("ennemy_3", 4750, self.fixed_height, 9000, 0, pi, 0)
        df.reset_machine_matrix("ennemy_4", 5250, self.fixed_height, 9000, 0, pi, 0)
        df.reset_machine_matrix("ennemy_5", 7750, self.fixed_height, 9000, 0, pi, 0)
        df.reset_machine_matrix("ennemy_6", 8250, self.fixed_height, 9000, 0, pi, 0)

        # 3-3 分布
        # df.reset_machine_matrix("ennemy_1", 1750, self.fixed_height, 9000, 0, pi, 0)
        # df.reset_machine_matrix("ennemy_2", 2250, self.fixed_height, 8500, 0, pi, 0)
        # df.reset_machine_matrix("ennemy_3", 2750, self.fixed_height, 9000, 0, pi, 0)
        # df.reset_machine_matrix("ennemy_4", 7250, self.fixed_height, 9000, 0, pi, 0)
        # df.reset_machine_matrix("ennemy_5", 7750, self.fixed_height, 8500, 0, pi, 0)
        # df.reset_machine_matrix("ennemy_6", 8250, self.fixed_height, 9000, 0, pi, 0)

    def set_human_preference(self, human_preference):
        """
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
        """
        self.human_preference = human_preference[0]

    def step(self, action_ally, attention_weight):
        self.show_plane_id()
        self.current_step += 1
        # self.attention_weight = attention_weight
        # 对TTA输出的attention_weight增加随机扰动
        # 随机选择一个元素，并将其增加0.5
        self.attention_weight = self._add_attention_weight_perturb(attention_weight)

        self._apply_action(action_ally)  # 执行动作

        self._get_plane_state()  # 双方所有飞机的状态信息分别存储到self.ally_state_n与self.enemy_state_n

        # 将权重映射为整数优先级，数字越大优先级越大
        int_attention_weight = self.int_attention_weight(self.human_preference)

        ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n = self._get_attention_observation(int_attention_weight)  # 获取状态观测

        rewards = self._get_reward(int_attention_weight)  # 获取当前状态奖励
        step_penalty = -0.005
        rewards += step_penalty  # 步数惩罚

        dones = self._get_done()  # 任务完成标志

        if all(dones):
            self.num_episode += 1
            self.sum_destroy_enemy += self.num_destroy_enemy
            self.sum_steps += self.current_step
            if self.num_episode > 0:
                print(f"总回合数：{self.num_episode}, 胜利回合数：{self.num_win}, 完全击毁率：{self.num_win/self.num_episode*100:.2f}%, "
                      f"击毁敌机数：{self.num_destroy_enemy}架, 平均击毁敌机数：{self.sum_destroy_enemy/self.num_episode:.2f}架, "
                      f"任务步数：{self.current_step}步, 平均任务步数：{self.sum_steps/self.num_episode:.2f}步")

        # 给前端UI发送数据
        data = self.data_for_UI(self.attention_weight, int_attention_weight)
        msg = json.dumps(data) + "\n"
        self.UIServer.send(msg)

        # 在step的最后将当前状态信息赋值给old，作为下一个step计算相邻时间状态距离差的输入
        self.old_ally_state_n = self.ally_state_n
        self.old_enemy_state_n = self.enemy_state_n

        df.update_scene()  # 场景更新

        return ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n, rewards, dones, {}

    def _apply_action(self, action_ally):
        # 执行己方动作
        norm_action = self.normalize_action(action_ally)  # 离散动作的处理，归一化为[-1,+1]之间
        for ally_id in range(self.num_allyUAVs):
            ally_state = self.ally_state_n[ally_id]
            if ally_state["health_level"] == 0:
                continue
            # 设置动作
            roll = norm_action[ally_id][0]
            pitch = norm_action[ally_id][1]
            yaw = norm_action[ally_id][2]
            df.set_plane_roll(self.Plane_ID_ally[ally_id], np.float64(roll))  # 滚转
            df.set_plane_pitch(self.Plane_ID_ally[ally_id], np.float64(pitch))  # 俯仰
            df.set_plane_yaw(self.Plane_ID_ally[ally_id], np.float64(yaw))  # 方向舵
            df.set_plane_linear_speed(self.Plane_ID_ally[ally_id], self.base_speed)  # 定速定的是原始线速度

        # 执行敌方动作
        for oppo_id in range(self.num_oppoUAVs):
            # oppo_state = df.get_plane_state(self.Plane_ID_oppo[oppo_id])
            oppo_state = self.enemy_state_n[oppo_id]
            if oppo_state["health_level"] == 0:
                continue
            else:
                df.set_plane_linear_speed(self.Plane_ID_oppo[oppo_id], self.base_speed - 180)  # 存活敌机设置速度

    def _get_reward(self, int_attention_weight):
        hit_prob = 1.0
        is_success = False
        self.reward = np.zeros((self.num_allyUAVs, 1), dtype=np.float64)

        for ally_id in range(self.num_allyUAVs):
            # 获取己方飞机状态
            ally_state = self.ally_state_n[ally_id]
            if ally_state["health_level"] == 0:
                continue

            # 己方飞机出界惩罚
            is_out = self.is_out(ally_state["position"])
            if is_out:
                self.reward[ally_id][0] -= 25.0
                df.set_health(self.Plane_ID_ally[ally_id], 0)
                # logging.info(self.Plane_ID_ally[ally_id] + " 出界...")
                continue

            # 初始化敌机奖励
            num_dead_oppo = 0
            reward_enemy = np.zeros((self.num_oppoUAVs,), dtype=np.float64)
            for oppo_id in range(self.num_oppoUAVs):
                oppo_state = self.enemy_state_n[oppo_id]
                if oppo_state["health_level"] == 0:
                    num_dead_oppo += 1
                    continue

                # 计算相对位置与距离
                relative_pos = np.array([oppo_state["position"][0] - ally_state["position"][0],
                                         oppo_state["position"][1] - ally_state["position"][1],
                                         oppo_state["position"][2] - ally_state["position"][2]])
                now_distance = np.linalg.norm(relative_pos)
                angle_2oppo = abs(self._angle_dist(ally_state["move_vector"], relative_pos))
                angle_2ally = abs(self._angle_dist(oppo_state["move_vector"], -relative_pos))

                # 1. 相对角度奖励
                angle_advantage = self.get_angle_advantage(angle_2oppo, np.pi - angle_2ally)  # [0, 1]
                reward_enemy[oppo_id] += angle_advantage * 0.1 * int_attention_weight[oppo_id][0]

                # 2. 攻击范围内奖励
                if angle_2oppo < self.attack_angle:
                    reward_enemy[oppo_id] += 0.2 * int_attention_weight[oppo_id][0]

                # 3. 成功击毁奖励
                if angle_2oppo < self.attack_angle and now_distance < self.attack_dist:  # 攻击条件
                    hit = random() < hit_prob
                    if hit:
                        # logging.info(self.Plane_ID_oppo[oppo_id] + " be killed...")
                        df.set_health(self.Plane_ID_oppo[oppo_id], 0)
                        reward_enemy[oppo_id] += 10.0 * int_attention_weight[oppo_id][0]
                    else:
                        reward_enemy[oppo_id] += 1.0 * int_attention_weight[oppo_id][0]

                # 4. 己方被攻击惩罚
                if angle_2ally < self.attack_angle and now_distance < self.attack_dist:
                    hit = random() < hit_prob
                    if hit:
                        # logging.info(self.Plane_ID_ally[ally_id] + " be killed...")
                        df.set_health(self.Plane_ID_ally[ally_id], 0)
                        reward_enemy[oppo_id] -= 25.0
                    else:
                        reward_enemy[oppo_id] -= 5.0

            reward_enemy_fused = np.sum(reward_enemy)  # 直接求和
            self.reward[ally_id][0] += reward_enemy_fused

            self.num_destroy_enemy = num_dead_oppo
            # 敌方全部击毁奖励
            if num_dead_oppo == self.num_oppoUAVs:
                is_success = True
                self.reward[ally_id][0] += 200.0

        if is_success:
            self.num_win += 1
            print("success: all enemies dead")

        return self.reward

    def _get_done(self):
        self.dones = np.full((self.num_allyUAVs, 1), False, dtype=bool)
        self.oppo_dones = np.full((self.num_oppoUAVs, 1), False, dtype=bool)

        ally_destroyed = [False for _ in range(self.num_allyUAVs)]
        oppo_destroyed = [False for _ in range(self.num_oppoUAVs)]

        for i in range(self.num_UAVs):
            # ally_state = df.get_plane_state(self.Plane_ID_ally[i])
            # oppo_state = df.get_plane_state(self.Plane_ID_oppo[i])
            ally_state = self.ally_state_n[i]
            oppo_state = self.enemy_state_n[i]

            # constraint combat range [10000*10000], height is limited in range(1000,8000)
            self.dones[i][0] = self.is_out(ally_state["position"])
            self.oppo_dones[i][0] = self.is_out(oppo_state["position"])

            if ally_state["health_level"] == 0:
                self.dones[i][0] = True
                ally_destroyed[i] = True
            if oppo_state["health_level"] == 0:
                self.oppo_dones[i] = True
                oppo_destroyed[i] = True

        if all(ally_destroyed) or all(oppo_destroyed):
            self.dones = np.full((self.num_allyUAVs, 1), True, dtype=bool)

        # 我方飞机全部done==true重置环境的功能写在env_wrappers里面
        if all(self.oppo_dones):
            self.dones = np.full((self.num_allyUAVs, 1), True, dtype=bool)

        return self.dones

    def _get_attention_observation(self, int_attention_weight):
        # Plane States
        ally_obs_n = []
        ally_attention_n = np.zeros((self.num_allyUAVs, 7), dtype=np.float64)
        enemy_attention_n = np.zeros((self.num_oppoUAVs, 7), dtype=np.float64)

        for i in range(self.num_allyUAVs):
            own_feature = np.zeros((7,), dtype=np.float64)  # 自身特征
            fri_feature = np.zeros((self.num_allyUAVs - 1, 7), dtype=np.float64)  # 队友特征
            ennemy_feature = np.zeros((self.num_oppoUAVs, 11), dtype=np.float64)  # 敌人特征
            fri_alive_mask = np.zeros((self.num_allyUAVs - 1,), dtype=np.float64)  # 队友存活编码
            ennemy_alive_mask = np.zeros((self.num_oppoUAVs,), dtype=np.float64)  # 敌人存活编码

            # own_state = df.get_plane_state(self.Plane_ID_ally[i])  # TCP通信获取状态
            own_state = self.ally_state_n[i]  # 使用事先存储好的状态，不需要每次都进行TCP通信损耗时间

            if own_state["health_level"] > 0:
                # 自身位置信息
                own_feature[0:3] = np.array([own_state["position"][0] / NormStates["Plane_position"],
                                             own_state["position"][1] / NormStates["Plane_position"],
                                             own_state["position"][2] / NormStates["Plane_position"]])
                # 自身速度信息
                own_feature[3:6] = np.array([own_state["move_vector"][0] / NormStates["Plane_move_vector"],
                                             own_state["move_vector"][1] / NormStates["Plane_move_vector"],
                                             own_state["move_vector"][2] / NormStates["Plane_move_vector"]])

                own_feature[6] = own_state["health_level"]  # 飞机健康值

                fri_idx = 0
                for friend in range(self.num_allyUAVs):
                    if friend == i:  # 遍历到自己则跳过操作
                        continue
                    # fri_idx_state = df.get_plane_state(self.Plane_ID_ally[friend])  # TCP通信获取状态
                    fri_idx_state = self.ally_state_n[friend]  # 使用事先存储好的状态，不需要每次都进行TCP通信损耗时间
                    if fri_idx_state["health_level"] > 0:
                        relative_pos = np.array([fri_idx_state["position"][0] - own_state["position"][0],
                                                 fri_idx_state["position"][1] - own_state["position"][1],
                                                 fri_idx_state["position"][2] - own_state["position"][2]])
                        relative_vel = np.array([fri_idx_state["move_vector"][0] - own_state["move_vector"][0],
                                                 fri_idx_state["move_vector"][1] - own_state["move_vector"][1],
                                                 fri_idx_state["move_vector"][2] - own_state["move_vector"][2]])
                        distance = np.linalg.norm(relative_pos)

                        fri_feature[fri_idx, 0:3] = relative_pos / 2000  # 队友相对位置矢量
                        fri_feature[fri_idx, 3] = distance / 2000  # 队友相对自己的距离
                        fri_feature[fri_idx, 4:7] = relative_vel / NormStates["Plane_move_vector"]  # 队友相对速度

                        fri_alive_mask[fri_idx] = 1  # 表示该队友存活
                    else:
                        fri_alive_mask[fri_idx] = 0  # 表示该队友被击毁

                    fri_idx += 1

                for ennemy_idx in range(self.num_oppoUAVs):
                    enemy_attention_feature = np.zeros((7,), dtype=np.float64)  # 用于注意力机制的敌人特征
                    # ennemy_idx_state = df.get_plane_state(self.Plane_ID_oppo[ennemy_idx])  # TCP通信获取状态
                    ennemy_idx_state = self.enemy_state_n[ennemy_idx]  # 使用事先存储好的状态，不需要每次都进行TCP通信损耗时间
                    if ennemy_idx_state["health_level"] > 0:
                        # 己方状态默认保持真实观测
                        own_pos = np.asarray(own_state["position"], dtype=np.float64)
                        own_vel = np.asarray(own_state["move_vector"], dtype=np.float64)

                        # 敌方真实状态
                        enemy_pos_true = np.asarray(ennemy_idx_state["position"], dtype=np.float64)
                        enemy_vel_true = np.asarray(ennemy_idx_state["move_vector"], dtype=np.float64)

                        # 敌方位置观测时延：只作用于观测，不修改真实环境状态
                        enemy_pos_delay = self._add_enemy_pos_delay(enemy_pos_true)

                        # 敌方状态只在观测层面加随机噪声
                        enemy_pos_obs = self._add_enemy_obs_noise(enemy_pos_delay)
                        enemy_vel_obs = self._add_enemy_obs_noise(enemy_vel_true)

                        # relative_pos = np.array([ennemy_idx_state["position"][0] - own_state["position"][0],
                        #                          ennemy_idx_state["position"][1] - own_state["position"][1],
                        #                          ennemy_idx_state["position"][2] - own_state["position"][2]])
                        # relative_vel = np.array([ennemy_idx_state["move_vector"][0] - own_state["move_vector"][0],
                        #                          ennemy_idx_state["move_vector"][1] - own_state["move_vector"][1],
                        #                          ennemy_idx_state["move_vector"][2] - own_state["move_vector"][2]])

                        # 用带噪声的敌方观测计算相对位置、相对速度、距离和角度
                        relative_pos = enemy_pos_obs - own_pos
                        relative_vel = enemy_vel_obs - own_vel

                        distance = np.linalg.norm(relative_pos)
                        att_angle = self._angle_dist(own_state["move_vector"], relative_pos)  # 我对敌的攻击角
                        angle_2own = self._angle_dist(enemy_vel_obs, -relative_pos)  # 敌对我的攻击角
                        angle_vel = self._angle_dist(own_state["move_vector"], enemy_vel_obs)  # 速度方向夹角

                        ennemy_feature[ennemy_idx, 0:3] = relative_pos / 5000  # 敌方相对位置矢量
                        ennemy_feature[ennemy_idx, 3] = distance / 5000  # 敌方相对自己的距离
                        ennemy_feature[ennemy_idx, 4:7] = relative_vel / NormStates["Plane_move_vector"]  # 敌方相对速度
                        ennemy_feature[ennemy_idx, 7] = att_angle / pi  # 攻击角
                        ennemy_feature[ennemy_idx, 8] = (pi - angle_2own) / pi  # 逃跑角
                        ennemy_feature[ennemy_idx, 9] = angle_vel  # 速度方向夹角
                        ennemy_feature[ennemy_idx, 10] = int_attention_weight[ennemy_idx][0]  # 优先级

                        ennemy_alive_mask[ennemy_idx] = 1

                        # enemy feature for attention
                        # ennemy_idx对应的敌机位置
                        # enemy_attention_feature[0:3] = np.array(
                        #     [ennemy_idx_state["position"][0] / NormStates["Plane_position"],
                        #      ennemy_idx_state["position"][1] / NormStates["Plane_position"],
                        #      ennemy_idx_state["position"][2] / NormStates["Plane_position"]])
                        # # ennemy_idx对应的敌机速度矢量
                        # enemy_attention_feature[3:6] = np.array(
                        #     [ennemy_idx_state["move_vector"][0] / NormStates["Plane_move_vector"],
                        #      ennemy_idx_state["move_vector"][1] / NormStates["Plane_move_vector"],
                        #      ennemy_idx_state["move_vector"][2] / NormStates["Plane_move_vector"]])

                        # enemy feature for attention，同样使用带噪声的敌方观测
                        enemy_attention_feature[0:3] = enemy_pos_obs / NormStates["Plane_position"]
                        enemy_attention_feature[3:6] = enemy_vel_obs / NormStates["Plane_move_vector"]
                        enemy_attention_feature[6] = ennemy_idx_state["health_level"]
                    else:
                        ennemy_alive_mask[ennemy_idx] = 0

                    # 存储每一个敌方的注意力机制特征信息
                    enemy_attention_n[ennemy_idx] = enemy_attention_feature

            # 存储每一个己方的注意力机制特征信息
            ally_attention_n[i] = own_feature

            ally_obs = np.concatenate([own_feature.flatten(),
                                       fri_feature.flatten(),
                                       ennemy_feature.flatten(),
                                       fri_alive_mask.flatten(),
                                       ennemy_alive_mask.flatten()
                                       ])
            ally_obs_n.append(ally_obs)

        ally_obs_n = np.stack(ally_obs_n)
        ally_share_obs = np.array([np.concatenate(ally_obs_n, axis=0).copy() for _ in range(self.num_allyUAVs)])

        # 己方注意力特征与敌方注意力特征
        ally_attention_n = np.array(ally_attention_n)
        enemy_attention_n = np.array(enemy_attention_n)

        return ally_obs_n, ally_share_obs, ally_attention_n, enemy_attention_n

    def _add_enemy_pos_delay(self, enemy_position):
        """
        给敌方位置观测增加时延。

        当前环境假设：
            1. 敌方单位沿三维坐标最后一维匀速运动；
            2. 敌方速度恒定为 120 m/s；
            3. 高度位置不变；
            4. 该函数只修改观测位置，不修改真实环境状态。

        因此：
            delayed_z = true_z - 120 * delay_time
        """
        enemy_position = np.asarray(enemy_position, dtype=np.float64).copy()

        if (not self.enable_enemy_obs_delay) or self.enemy_obs_delay_sec <= 0:
            return enemy_position

        delay_distance = self.enemy_obs_delay_speed * self.enemy_obs_delay_sec
        enemy_position[2] -= delay_distance

        return enemy_position

    def _add_enemy_obs_noise(self, value):
        """
        给敌方观测量增加 5% 乘性随机噪声。
        形式：
            x_obs = x_true * (1 + epsilon)
            epsilon ~ Uniform(-noise_ratio, noise_ratio)
        例如 noise_ratio=0.05 时，表示每个维度最多产生 ±5% 的随机扰动。
        """
        value = np.asarray(value, dtype=np.float64)

        if (not self.enable_enemy_obs_noise) or self.enemy_obs_noise_ratio <= 0:
            return value.copy()

        # 如果已经通过 env.seed(seed) 设置了随机种子，则使用 self.np_random；
        # 否则使用 np.random，避免 __init__ 中 reset() 早于 seed() 导致报错。
        rng = self.np_random if self.np_random is not None else np.random

        noise = rng.uniform(
            low=-self.enemy_obs_noise_ratio,
            high=self.enemy_obs_noise_ratio,
            size=value.shape
        )

        return value * (1.0 + noise)

    def _add_attention_weight_perturb(self, attention_weight):
        """
        对TTA输出的attention_weight增加随机扰动。

        具体方式：
            随机选择attention_weight中的一个元素，并将其数值增加0.5。

        注意：
            1. 该函数只扰动self.attention_weight；
            2. 不修改输入attention_weight本身；
            3. 不修改真实环境状态。
        """
        attention_weight = np.asarray(attention_weight, dtype=np.float64).copy()

        if not self.enable_attention_weight_perturb:
            return attention_weight

        flat_weight = attention_weight.reshape(-1)

        # 使用环境随机种子，保证可复现；如果没有seed，则使用np.random
        rng = self.np_random if self.np_random is not None else np.random

        perturb_idx = rng.integers(0, flat_weight.shape[0])
        flat_weight[perturb_idx] += self.attention_weight_perturb_value

        return flat_weight.reshape(attention_weight.shape)

    def is_out(self, position):
        if position[0] > 10000 or position[0] < 0 or \
           position[1] > 8000 or position[1] < 1000 or \
           position[2] > 10000 or position[2] < 0:
            return True
        else:
            return False

    def get_3d_angle(self, relative_pos):
        x = relative_pos[0]
        y = relative_pos[2]
        z = relative_pos[1]

        # 方位角
        phi = atan2(y, x)
        # 俯仰角
        theta = atan2(z, math.sqrt(x**2 + y**2))

        return phi, theta

    def _last_distance(self):
        """
        Returns:
            last_distance_n: 一个二维数组，存储每一个己方飞机与每一个敌方飞机的距离
            last_distance_sum: 所有距离之和
        """
        last_distance_n = [[] for _ in range(self.num_allyUAVs)]
        last_distance_sum = 0
        for ally_id in range(self.num_allyUAVs):
            ally_state = self.old_ally_state_n[ally_id]
            if ally_state["health_level"] == 0:
                last_distance_n[ally_id] = [0 for _ in range(self.num_oppoUAVs)]
                continue

            for oppo_id in range(self.num_oppoUAVs):
                oppo_state = self.old_enemy_state_n[oppo_id]
                if oppo_state["health_level"] == 0:
                    last_distance_n[ally_id].append(0)
                    continue
                relative_pos = np.array([ally_state["position"][0] - oppo_state["position"][0],
                                         ally_state["position"][1] - oppo_state["position"][1],
                                         ally_state["position"][2] - oppo_state["position"][2]])
                distance = np.linalg.norm(relative_pos)
                height_diff = abs(ally_state["position"][1] - oppo_state["position"][1])  # 高度差

                last_distance_n[ally_id].append(distance)
                last_distance_sum += distance

        return last_distance_sum, last_distance_n

    @staticmethod
    def _angle_dist(own_toward, relative_pos):
        """
        Args:
            own_toward: 自身朝向
            relative_pos: 相对位置向量

        Returns:
            angle_radians: 自身朝向与相对位置向量的夹角，正负区分左右
        """

        cross_product = np.cross(own_toward, relative_pos)
        dot_product = np.dot(own_toward, relative_pos)
        norm_a = np.linalg.norm(own_toward)
        norm_b = np.linalg.norm(relative_pos)
        cos_theta = dot_product / (norm_a * norm_b)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 避免由于数值误差导致溢出

        if norm_a == 0 or norm_b == 0:
            return 0

        # left or right
        if cross_product[2] < 0:
            angle_radians = -angle_radians

        # return relative angle between 2 planes
        return angle_radians

    def get_angle_advantage(self, angle_att, angle_esp):
        """
        Args:
            angle_att: 攻击角
            angle_esp: 逃跑角
        Returns: 攻击方在逃跑方的身后才有优势
        """
        factor_1 = (pi - angle_att) / pi
        factor_2 = (pi - angle_esp) / pi

        value = factor_1 * factor_2

        return value

    def normalize_action(self, actions):
        """
        为离散动作设计，该函数表示对每个ally_id的离散动作归一化转换。
        # 俯仰：离散化为n份 (n = 3)
        # 偏航：离散化为n份
        # 滚转：离散化为n份
        # 速度：离散化为n份
        """
        norm_acts = np.zeros(actions.shape)  # 初始化一个和输入动作同形状的数组
        for i in range(actions.shape[1]):  # 遍历每一列动作
            # 正确地访问 Discrete 对象的 n 属性
            action_range = self.ally_action_spaces[i].n if hasattr(self.ally_action_spaces[i], 'n') else \
                self.ally_action_spaces[i]
            norm_acts[:, i] = (actions[:, i] * 2.0 / (action_range - 1) - 1)

        # 应用最大值缩放
        norm_acts[:, 0] *= 1  # 滚转
        norm_acts[:, 1] *= 1  # 俯仰
        norm_acts[:, 2] *= 1  # 偏航

        return norm_acts

    def int_attention_weight(self, human_preference):
        # 将 attention_weight 转换为 NumPy 数组，并压缩为 1D
        attention_weight = np.array(self.attention_weight).flatten()  # 转换为 [0.2, 0.8, 0.5, 0.1]
        attention_weight += human_preference

        # 使用 `argsort` 两次实现稳定排名
        sorted_indices = np.argsort(attention_weight)  # 从小到大排序的索引
        ranks = np.zeros_like(attention_weight, dtype=int)
        rank = 1

        for i in range(len(sorted_indices)):
            if i > 0 and attention_weight[sorted_indices[i]] != attention_weight[sorted_indices[i - 1]]:
                rank += 1
            ranks[sorted_indices[i]] = rank

        # 将排名结果转回为二维形式，并更新 self.attention_weight
        return ranks.reshape(-1, 1).tolist()

    def int_attention_weight_v2(self, human_preference):
        # 将 attention_weight 转换为 NumPy 数组，并压缩为 1D
        attention_weight = np.array(self.attention_weight).flatten()  # 转换为 [0.2, 0.8, 0.5, 0.1]
        attention_weight += human_preference

        for i in range(self.num_oppoUAVs):
            if self.enemy_state_n[i]["health_level"] == 0:
                attention_weight[i] = -np.inf

        # 使用 `argsort` 两次实现稳定排名
        sorted_indices = np.argsort(attention_weight)[::-1]  # 从大到小排序的索引
        ranks = np.zeros_like(attention_weight, dtype=int)

        rank = self.num_oppoUAVs
        for i in range(len(sorted_indices)):
            if attention_weight[sorted_indices[i]] != -np.inf:
                ranks[sorted_indices[i]] = rank
                rank -= 1

        # 将排名结果转回为二维形式，并更新 self.attention_weight
        return ranks.reshape(-1, 1).tolist()

    def reset_after_hit(self, plane_id):
        """
        飞机被击毁后，reset到一个指定位置
        """
        df.reset_machine_matrix(plane_id, 0, 4000, -100, 0, 0, 0)
        df.set_plane_linear_speed(plane_id, 0)

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def show_plane_id(self):
        for ally_id in range(self.num_allyUAVs):
            ally_state = df.get_plane_state(self.Plane_ID_ally[ally_id])
            if ally_state["health_level"] == 0:
                continue
            df.display_vector(ally_state["position"], ally_state["move_vector"], self.Plane_ID_ally[ally_id],
                              [0, 0.02], [0, 1, 0, 1], label_size=0.02)
        for oppo_id in range(self.num_oppoUAVs):
            oppo_state = df.get_plane_state(self.Plane_ID_oppo[oppo_id])
            if oppo_state["health_level"] == 0:
                continue
            # print(oppo_state["position"])
            df.display_vector(oppo_state["position"], oppo_state["move_vector"], self.Plane_ID_oppo[oppo_id],
                              [0, 0.02], [1, 0, 0, 1], label_size=0.02)

    def data_for_UI(self, attention_weight, mixed_priority):
        allies = [
            {
                "id": f"ally{i + 1}",
                "x": self.ally_state_n[i]["position"][0],
                "y": self.ally_state_n[i]["position"][1],
                "z": self.ally_state_n[i]["position"][2],
                "vx": self.ally_state_n[i]["move_vector"][0],
                "vy": self.ally_state_n[i]["move_vector"][1],
                "vz": self.ally_state_n[i]["move_vector"][2],
                "rotation": {
                    "yaw": math.degrees(self.ally_state_n[i]["Euler_angles"][1]),
                    "pitch": math.degrees(self.ally_state_n[i]["Euler_angles"][0]),
                    "roll": math.degrees(self.ally_state_n[i]["Euler_angles"][2])
                },
                "health_level": self.ally_state_n[i]["health_level"]
            }
            for i in range(self.num_allyUAVs)
        ]
        enemies = [
            {
                "id": f"enemy{i + 1}",
                "x": self.enemy_state_n[i]["position"][0],
                "y": self.enemy_state_n[i]["position"][1],
                "z": self.enemy_state_n[i]["position"][2],
                "vx": self.enemy_state_n[i]["move_vector"][0],
                "vy": self.enemy_state_n[i]["move_vector"][1],
                "vz": self.enemy_state_n[i]["move_vector"][2],
                "rotation": {
                    "yaw": math.degrees(self.enemy_state_n[i]["Euler_angles"][1]),
                    "pitch": math.degrees(self.enemy_state_n[i]["Euler_angles"][0]),
                    "roll": math.degrees(self.enemy_state_n[i]["Euler_angles"][2])
                },
                "health_level": self.enemy_state_n[i]["health_level"]
            }
            for i in range(self.num_oppoUAVs)
        ]

        # 处理attention_weight数据形式[[0, 0, 0, 0, 0, 0]]
        attention_weight = attention_weight.tolist()
        agent_preference = [sum(attention_weight, [])]
        # 同理处理mixed_priority的数据形式
        mixed_priority = [sum(mixed_priority, [])]

        data = {
            "type": "update_positions",
            "ally": allies,
            "enemy": enemies,
            "agent_preference": agent_preference,
            "mixed_priority": mixed_priority
        }

        return data
