# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/1/19 13:03

from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
from spot_robot import Robot
from trajectory_generator import Bezier
from spot_leg import Leg
import time



class Spot_gym(gym.Env):
    def __init__(self, render: bool = False, number_motor=12):
        self.render_flag = render
        self.spot_leg = Leg()
        if self.render_flag:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.robot = self._pybullet_client.loadURDF("../urdf/spot_old_2.urdf",[0,0,0.3],
                                                   useMaximalCoordinates=False,
                                                   flags=self._pybullet_client.URDF_USE_IMPLICIT_CYLINDER,
                                                   baseOrientation=self._pybullet_client.getQuaternionFromEuler(
                                                        [0, 0, np.pi])
                                                   )
        self.planeID = self._pybullet_client.loadURDF("plane.urdf")

        self.spot = Robot(self.robot, self._pybullet_client)
        self.action_bound = 1
        # action_space
        action_high = np.array([self.action_bound] * 12)
        self.action_space = spaces.Box(
            low=-action_high, high=action_high,
            dtype=np.float32)
        # observation space
        observation_high = self.spot.get_observation_upper_bound()
        self.observation_space = spaces.Box(
            low=-observation_high, high=observation_high,
            dtype=np.float64)

        self.tg = Bezier()

        self.control_frequency = 40
        self.dt = 1./self.control_frequency  # should be related to leg control frequency

        self.forward_weightX = 0.015
        self.forward_weightY = 0.01
        self.forwardV_weight = 0.01
        self.direction_weight = -0.001
        self.shake_weight = -0.005
        self.height_weight = -0.05
        self.joint_weight = -0.001

        self.angleFromReferen = np.array([0] * 12)

        self.pre_coorX  = 0.05383
        self.pre_height = 0.1933
        #
        # optimize signal
        self.opti_shoulder = np.deg2rad(5)
        self.opti_kneeAhid = np.deg2rad(20)
        self.referSignal = 1.
        self.optimSignal = 0.5

        self.reward_details = np.array([0.] * 5, dtype=np.float32)
        self.reward_detail_dict = {'forwardX': 0, 'forwardY': 0, 'forwardV_reward': 0, 'shaking_reward': 0,
                                   'height_reward': 0}
        self.step_num = 0
        self.initial_count = 0


        # 0.05382755820485801, 0, 0.19330842049777203




    def reset(self):
        # ----------initialize pubullet env----------------
        # self._pybullet_client.resetSimulation()
        self._pybullet_client.setGravity(0, 0, 0)


        # self.robot = self._pybullet_client.loadURDF("../urdf/spot_old_2.urdf", [0, 0, 0.3],
        #                                             useMaximalCoordinates=False,
        #                                             flags=p.URDF_USE_IMPLICIT_CYLINDER,
        #                                             baseOrientation=self._pybullet_client.getQuaternionFromEuler(
        #                                                 [0, 0, np.pi])
        #                                             )
        self._pybullet_client.resetBasePositionAndOrientation(bodyUniqueId=self.robot, posObj=[0, 0, 0.3],
                                                            ornObj=self._pybullet_client.getQuaternionFromEuler([0, 0, np.pi] ) )
        self._pybullet_client.changeDynamics(bodyUniqueId=self.robot, linkIndex=-1, mass=1.5)
        self.spot = Robot(self.robot, self._pybullet_client)

        #  ----------------------------------initial parameter------------------#
        self.reward_details = np.array([0.] * 5, dtype=np.float32)
        self.step_num = 0
        self.spot_leg.time_reset()
        self.pre_coorX  = 0.05383
        self.pre_height = 0.1933
        #  ----------------------------------initial parameter------------------#

        while self.initial_count < 100:
            self._pybullet_client.setGravity(0, 0, 0)
            self.initial_count += 1
            self.spot_leg.positions_control2(self.robot, self.spot.stand_pose[0], self.spot.stand_pose[1],
                                                         self.spot.stand_pose[2], self.spot.stand_pose[3])
            self.spot.motor_angle = np.hstack((self.spot.stand_pose))
            p.stepSimulation()

        while self.initial_count < 300:
            self._pybullet_client.setGravity(0, 0, -9.8)
            self.initial_count += 1
            self.spot_leg.positions_control2(self.robot, self.spot.stand_pose[0], self.spot.stand_pose[1],
                                                         self.spot.stand_pose[2], self.spot.stand_pose[3])
            self.spot.motor_angle = np.hstack((self.spot.stand_pose))
            p.stepSimulation()
        self.initial_count = 0
        return self.get_observation()


    def get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in ")
        observation = self.spot.get_observation()
        return observation

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        if self.physics_client_id >= 0:
            self._pybullet_client.disconnect()
        self.physics_client_id = -1


    def merge_action(self, action):
        LF = [0, 0, 0]
        RF = [0, 0, 0]
        LB = [0, 0, 0]
        RB = [0, 0, 0]

        # shoulder optimize signal from -3° to 3 °
        LF[0] = action[0] * self.opti_shoulder
        RF[0] = action[3] * self.opti_shoulder
        LB[0] = action[6] * self.opti_shoulder
        RB[0] = action[9] * self.opti_shoulder

        # hip,knee optimize signal from -15° to 15 °
        LF[1:] = action[1:3] * self.opti_kneeAhid
        RF[1:] = action[4:6] * self.opti_kneeAhid
        LB[1:] = action[7:9] * self.opti_kneeAhid
        RB[1:] = action[10:] * self.opti_kneeAhid
        return np.hstack((LF, RF, LB, RB)) * self.optimSignal + self.angleFromReferen * self.referSignal


    def apply_action(self, action):
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in")

        # action_on_motor =self.merge_action(action)
        # self.leg.positions_control(self.robot, action_on_motor[0:3], action_on_motor[3:6],
        #                           action_on_motor[6:9], action_on_motor[9:12])

        if self.step_num >= 0 and self.step_num <= 20:
            random_force = np.random.uniform(-5, 5, 3)
            self._pybullet_client.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1,
                                                     forceObj=[random_force[0], random_force[1], random_force[2]],
                                                     posObj=[0.05383, 0, 0.1933],
                                                     flags=self._pybullet_client.WORLD_FRAME)

        x1, y1, z1 = self.tg.curve_generator(self.spot_leg.t1)
        x2, y2, z2 = self.tg.curve_generator(self.spot_leg.t2)
        theta1_1, theta1_2 ,theta1_3 = self.spot_leg.IK_L_2(x1, y1,  z1)
        theta2_1, theta2_2 ,theta2_3 = self.spot_leg.IK_L_2(x2, y2,  z2)

        self.angleFromReferen = np.array([theta1_1, theta1_2 ,theta1_3, theta2_1, theta2_2 ,theta2_3,
                                          theta2_1, theta2_2 ,theta2_3, theta1_1, theta1_2 ,theta1_3 ])
        action_on_motor = self.merge_action(action)
        self.spot_leg.positions_control2(self.robot, action_on_motor[0:3], action_on_motor[3:6],
                                          action_on_motor[6:9], action_on_motor[9:12])
        self.spot.motor_angle = action_on_motor
        # ---------------test for free control------------------------#
        # self.woofh_leg.positions_control2( self.robot, [0, theta2 ,theta3], [0,theta4, theta5],
        #                              [0,theta4, theta5], [0, theta2 ,theta3])
        self.spot_leg.t1 += self.dt
        self.spot_leg.t2 += self.dt

    def reward_function(self,reward_items):
        x_coor  = reward_items[0]
        y_coor  = reward_items[1]
        linearX = reward_items[2]
        linearY, linearZ = reward_items[3:5]
        Wx, Wy, Wz   = reward_items[5:8]
        roll , pitch = reward_items[8:10]
        height       = reward_items[10]

        forwardX_reward = self.forward_weightX * (x_coor - self.pre_coorX)
        forwardY_reward = -self.forward_weightY * np.abs(y_coor)
        forwardV_reward = self.forwardV_weight * linearX / 4
        shaking_reward = self.shake_weight *     ( np.exp( -1/(Wx**2+Wy**2+Wz**2+ 1e-10)))/5+ \
                         self.shake_weight* ( roll**2 + pitch**2)
        height_reward = self.height_weight * (np.abs(height - self.pre_height))


        reward_details = np.array(
            [forwardX_reward, forwardY_reward, forwardV_reward, shaking_reward, height_reward])
        self.reward_details += reward_details
        reward = np.sum(reward_details)

        if self.step_num % 100 == 0:
            self.pre_coorX = x_coor
        self.pre_height = height

        return  reward



    def step(self, action):
        self.apply_action(action)
        self._pybullet_client.stepSimulation()
        self.step_num += 1
        state = self.get_observation()
        # print(f'state==={state}')
        reward_items = self.spot.get_reward_items()
        reward = self.reward_function(reward_items)
        roll, pitch, yaw = self.spot.get_ori()
        y = self.spot.get_Global_Coor()[1]
        # condition for stop
        if self.step_num > 1000:
            done = True
        elif roll > np.deg2rad(60) or pitch > np.deg2rad(60) or yaw > np.deg2rad(60) or y > 1.:
            reward = -10
            done = True
        else:
            done = False
        info = {}
        return state, reward, done, info

    def return_reward_details(self):
        self.reward_detail_dict['forwardX'] = self.reward_details[0]
        self.reward_detail_dict['forwardY'] = self.reward_details[1]
        self.reward_detail_dict['forwardV_reward'] = self.reward_details[2]
        self.reward_detail_dict['shaking_reward'] = self.reward_details[3]
        self.reward_detail_dict['height_reward'] = self.reward_details[4]



    def test_no_RL(self,model, test_round,test_speed):
        done = False
        self.optimSignal = 0
        all_episode_reward = []
        for i in range(test_round):
            obs = self.reset()
            episode_reward = 0
            while True:
                time.sleep(test_speed)
                action = model.predict(obs)
                obs, reward, done, _ = self.step(action[0])
                episode_reward += reward
                if done:
                    break
            print(f'episode_reward=={episode_reward}')
            self.return_reward_details()
            print(self.reward_detail_dict)
            all_episode_reward.append(episode_reward)
            print(f'all reward_episode is {all_episode_reward}')
        return all_episode_reward


if __name__ == '__main__':
    from  stable_baselines3 import PPO
    from  stable_baselines3.common.env_checker import check_env


    env = Spot_gym(render=True)

    model = PPO(policy="MlpPolicy", env=env, verbose=1)

    env.test_no_RL(model,10,0.0)
