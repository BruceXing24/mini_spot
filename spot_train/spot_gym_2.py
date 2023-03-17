# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/1/19 13:03

# spot_gym_2 is used to training climb on a 10° degree
# action space is still action space

from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import numpy as np
from spot_robot_2 import Robot
from  CPGenerator import CPG
from spot_leg import Leg
import time
from terrain import Terrain


class Spot_gym(gym.Env):
    def __init__(self, render: bool = False, number_motor=12):
        self.render_flag = render
        self.spot_leg = Leg()
        if self.render_flag:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        self._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.robot = self._pybullet_client.loadURDF("../urdf/spot_mini_3.urdf",[0,0,0.3],
                                                   baseOrientation = self._pybullet_client.getQuaternionFromEuler([0, 0,-np.pi/2]),
                                                   flags=self._pybullet_client.URDF_USE_INERTIA_FROM_FILE,
                                                   # useFixedBase = 1
                                                   )
        self.planeID = self._pybullet_client.loadURDF("plane.urdf")
        self.slope = self._pybullet_client.loadURDF("../urdf/slope/slope.urdf",
                                                   [2.5, 0, 0.0],
                                                    # flags=p.URDF_USE_INERTIA_FROM_FILE,
                                                    useFixedBase=1
                                                    )
        self._pybullet_client.changeDynamics(self.slope, linkIndex=-1, lateralFriction=1.0)
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
        self.gait_generator = CPG(step_length=0.03,ground_clearance=0.03,ground_penetration=0.05,Tswing=0.3,Tstance=0.3,initial_x=0.0)


        self.control_frequency = 50
        self.dt = 1./self.control_frequency  # should be related to leg control frequency

        self.forward_weightX = 0.2
        self.forward_weightY = 0.03
        self.forwardV_weight = 0.01
        self.direction_weight = -0.001
        self.shake_weight = -0.005
        self.height_weight = -0.05
        self.joint_weight = -0.001

        self.angleFromReferen = np.array([0] * 12)

        self.pre_coorX  = 0.165
        self.pre_height = 0.194
        #
        # optimize signal
        self.opti_shoulder = np.deg2rad(5)
        self.opti_kneeAhid = np.deg2rad(15)
        self.referSignal = 1
        self.optimSignal = 1

        self.reward_details = np.array([0.] * 5, dtype=np.float32)
        self.reward_detail_dict = {'forwardX': 0, 'forwardY': 0, 'forwardV_reward': 0, 'shaking_reward': 0,
                                   'height_reward': 0}


        self.step_num = 0
        self.initial_count = 0
        self.train_steps = 0

        self.terrain_generator = Terrain("random")
        self.reset_terrain_flag = True
        # 0.05382755820485801, 0, 0.19330842049777203
        # self.terrain = Terrain("random").generate_terrain(self._pybullet_client,0.02)



    def reset(self):
        # ----------initialize pubullet env----------------
        # self._pybullet_client.resetSimulation()
        self._pybullet_client.setGravity(0, 0, 0)
        # if self.reset_terrain_flag ==True:
        #     self.reset_terrain()
        #     print("reset------------------------terrain")
        #     self.reset_terrain_flag = False

        self._pybullet_client.resetBasePositionAndOrientation(bodyUniqueId=self.robot, posObj=[0, 0, 0.3],
                                                            ornObj=self._pybullet_client.getQuaternionFromEuler([0, 0, -np.pi/2] ) )
        # self._pybullet_client.changeDynamics(bodyUniqueId=self.robot, linkIndex=-1, mass=3)
        self.spot = Robot(self.robot, self._pybullet_client)



        #  ----------------------------------initial parameter------------------#
        self.reward_details = np.array([0.] * 5, dtype=np.float32)
        self.step_num = 0
        self.pre_coorX  = 0.165
        self.pre_height = 0.194
        self.gait_generator.trot_timer = 0.
        #  ----------------------------------initial parameter------------------#

        while self.initial_count < 100:
            self._pybullet_client.setGravity(0, 0, 0)
            self.initial_count += 1
            self.spot_leg.positions_control2(self.robot, self.spot.stand_pose[0], self.spot.stand_pose[1],
                                                         self.spot.stand_pose[2], self.spot.stand_pose[3])
            self.spot.motor_angle = np.hstack((self.spot.stand_pose))
            self._pybullet_client.resetBasePositionAndOrientation(bodyUniqueId=self.robot, posObj=[0, 0, 0.3],
                                                                  ornObj=self._pybullet_client.getQuaternionFromEuler(
                                                                      [0, 0, -np.pi / 2]))

            self._pybullet_client.resetBaseVelocity(objectUniqueId=self.robot, linearVelocity=[0, 0, 0],
                                                    angularVelocity=[0, 0, 0])
            p.stepSimulation()


        while self.initial_count < 250:
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
        # print(f"observation_space === { observation }")


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
            random_force = np.random.uniform(-3, 3, 3)
            self._pybullet_client.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1,
                                                     forceObj=[random_force[0], random_force[1], random_force[2]],
                                                     posObj=[0.165, 0, 0.194],
                                                     flags=self._pybullet_client.WORLD_FRAME)

        angles = self.gait_generator.trot_generator(self.gait_generator.trot_timer, 'straight', 0.8)
        angles = angles * np.pi / 180.
        self.gait_generator.trot_timer += 2. / 100
        self.angleFromReferen = angles

        action_on_motor = self.merge_action(action)
        self.spot_leg.positions_control2(self.robot, action_on_motor[0:3], action_on_motor[3:6],
                                          action_on_motor[6:9], action_on_motor[9:12])
        self.spot.motor_angle = action_on_motor
        # ---------------test for free control------------------------#
        # self.woofh_leg.positions_control2( self.robot, [0, theta2 ,theta3], [0,theta4, theta5],
        #                              [0,theta4, theta5], [0, theta2 ,theta3])

    def reward_function(self,reward_items):
        x_coor  = reward_items[0]
        y_coor  = reward_items[1]
        linearX = reward_items[2]
        linearY, linearZ = reward_items[3:5]
        Wx, Wy, Wz   = reward_items[5:8]
        roll , pitch ,yaw = reward_items[8:11]
        height       = reward_items[11]

        forwardX_reward = self.forward_weightX * (x_coor - self.pre_coorX)
        forwardY_reward = -self.forward_weightY * np.abs(y_coor)
        forwardV_reward = self.forwardV_weight * linearX / 4
        shaking_reward = self.shake_weight *     ( np.exp( -1/(Wx**2+Wy**2+Wz**2+ 1e-10)))/5+ \
                         self.shake_weight* ( roll**2 + pitch**2 + yaw**2)
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
        self.train_steps+=1
        # if self.train_steps % 400000 ==0:
        #     self.reset_terrain()
        state = self.get_observation()
        # print(f'state==={state}')
        reward_items = self.spot.get_reward_items()


        reward = self.reward_function(reward_items)
        roll, pitch, yaw = self.spot.get_ori()
        x = self.spot.get_Global_Coor()[0]
        y = self.spot.get_Global_Coor()[1]


        # condition for stop
        xyz = self.spot.get_Global_Coor()
        # print(f'x,y,z===={xyz[0]}{xyz[1]}{xyz[2]}')
        # print(f'roll==={np.rad2deg(roll)},pitch=={np.rad2deg(pitch)},yaw==={np.rad2deg(yaw)}')

        if self.step_num > 1000:
            done = True
        elif np.abs(roll) > np.deg2rad(45) or np.abs(pitch) > np.deg2rad(45) or np.abs(yaw) > np.deg2rad(45) or y > 0.5 or x <-0.3:
            reward = -5
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
                # check observation space
                # print(f'obs=={obs}')
                # check rpy
                # ori=self.spot.get_ori()
                # print(f'ori=={ori}')
                episode_reward += reward
                if done:
                    break
            print(f'episode_reward=={episode_reward}')
            self.return_reward_details()
            print(self.reward_detail_dict)
            all_episode_reward.append(episode_reward)
            print(f'all reward_episode is {all_episode_reward}')
            print(f"train_steps=={self.train_steps}")
        return all_episode_reward


    def test_model(self, model, test_round,test_speed, ):
        all_episode_reward = []
        height_set = []
        for i in range(test_round):
            episode_reward = 0
            obs = self.reset()
            while True:
                time.sleep(test_speed)
                action = model.predict(obs)
                obs, reward, done, _ = self.step(action[0])
                episode_reward += reward
                if done:
                    break
            all_episode_reward.append(episode_reward)
            print("episode reward===={}".format(episode_reward))
            self.return_reward_details()
            print(self.reward_detail_dict)
        all_episode_reward = np.array(all_episode_reward)
        print("all_reward==={},average reward=={}".format(all_episode_reward,np.sum(all_episode_reward)/test_round ))
        return all_episode_reward ,height_set


    def train(self,model,save_path,training_steps):
        model.learn(training_steps)
        model.save(save_path)

    def reset_terrain(self):
        self._pybullet_client.removeBody(self.terrain)
        self.terrain = Terrain("random").generate_terrain(self._pybullet_client)
        # self._pybullet_client.changeVisualShape(self.terrain, -1, rgbaColor=[0.5, 0.5, 0, 0.5])
        # self._pybullet_client.loadTexture("texture/grass.png")
        self._pybullet_client.resetBasePositionAndOrientation(self.terrain, [0, 0, 0], [0, 0, 0, 1])

if __name__ == '__main__':
    from  stable_baselines3 import PPO
    from  stable_baselines3.common.env_checker import check_env


    env = Spot_gym(render=True)
    # # -----------------training---------------#
    # model = PPO(policy="MlpPolicy", env=env, verbose=1,batch_size=512,learning_rate= 3e-4)
    # model = PPO(policy="MlpPolicy", env=env, verbose=1,tensorboard_log="./result/",learning_rate= 3e-4)
    # t1 = time.time()
    # model.learn(2000000)
    # model.save('result/train_result_2m_slope_realobs')
    # t2 = time.time()
    # print(t2-t1)
    # -----------------training---------------#

    # # ------------------test for no rl--------#
    # model = PPO(policy="MlpPolicy", env=env, verbose=1, batch_size=512)
    # env.test_no_RL(model,5,0.01)
    # ------------------test for no rl--------#

    # ----------------multi thread------------#

    #-----------------multi thread------------#

    # -----------------test---------------#
    # loaded_model = PPO.load('result/train_result_2m_slope_realobs')
    # env.test_model(loaded_model,5,0.01)
    # -----------------test---------------#