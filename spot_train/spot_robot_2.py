# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/1/19 13:03

from spot_leg import Leg
import numpy as np

class Robot(Leg):
    def __init__(self, robot,pybullet_client,):
        super(Leg,self).__init__()
        self.motor_angle = np.array([0] * 12)
        self._foot_id_list = [6, 11, 16, 21]
        '''                     FL         FR       BL      BR'''
        self.motor_id_list = [2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 20]
        self.pybullet_client = pybullet_client
        self.robot = robot
        self.stand_pose = [[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2]]
        self.sit_pose = [[0,0,0],[0,0,0],[0,np.pi/3,-np.pi*2/3],[0,np.pi/3,-np.pi*2/3]]
        # self.observation = self.get_observation()

    def get_base_height(self):
        posi, _ = self.pybullet_client.getBasePositionAndOrientation(self.robot)
        return np.array(posi[2])

    def get_Global_Coor(self):
        posi, _ = self.pybullet_client.getBasePositionAndOrientation(self.robot)
        return np.array(posi)

    def get_ori(self):
        _, ori = self.pybullet_client.getBasePositionAndOrientation(self.robot)
        ori = list(self.pybullet_client.getEulerFromQuaternion(ori))
        ori[2] = ori[2]+np.pi/2
        # calibration as urdf direction need to be turned pi/2
        return ori

    def get_linearV(self):
        linear_V, _ = self.pybullet_client.getBaseVelocity(self.robot)
        return linear_V

    def get_angularV(self):
        _, angularV = self.pybullet_client.getBaseVelocity(self.robot)
        return angularV

    def get_motor_angle(self):
        motor_angle = np.abs(self.motor_angle)
        return motor_angle


    def get_reward_items(self):
        x_coor = self.get_Global_Coor()[0]   # id 0
        y_coor = self.get_Global_Coor()[1]   #    1
        linearVxyz = self.get_linearV()      #    2 3 4
        angulerWxyz =self.get_angularV()     #    5 6 7
        rpy = self.get_ori()[0:3]            #    8 9 10
        height = self.get_base_height()      #    11
        return np.hstack((x_coor, y_coor, linearVxyz, angulerWxyz, rpy, height))

    def get_observation(self):
        #   3               3              3
        # r, p,y   ,  linear X,Y,Z,     angularX, Y, Z,
        rpy   =  self.get_ori()[0:3]
        angularXyz   =  self.get_angularV()
        height = self.get_base_height()
        joints_angle = self.get_motor_angle()
        return np.hstack((rpy,angularXyz, height,joints_angle))


    def get_observation_dim(self):
        return len(self.get_observation())


    def get_observation_upper_bound(self):
        upper_bound = np.array([0.0] * self.get_observation_dim())
        upper_bound[0:3] = np.pi
        upper_bound[3:6] = np.inf
        upper_bound[6] = 0.5
        upper_bound[7:] = np.pi

        return upper_bound