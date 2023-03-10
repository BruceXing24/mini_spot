# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/1/12 21:27

import pybullet as p
import pybullet_data as pd
from pybullet_utils import bullet_client
import numpy as np
import time
from spot_leg import Leg
from spot_train.trajectory_generator import Bezier

np.set_printoptions(suppress=True)

class Spot:
    def __init__(self):
        self.pybullet_client = bullet_client.BulletClient(connection_mode = p.GUI)
        self.robot = self.pybullet_client.loadURDF("urdf/spot_old.urdf",[0,0,0.3],
                                                   useMaximalCoordinates=False,
                                                   flags=self.pybullet_client.URDF_USE_IMPLICIT_CYLINDER,
                                                   )
        self.counter = 0
        self.show_fre = 0.005
        self.leg_controller = Leg()
        self.stand_pose = [[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2],[0,np.pi/4,-np.pi/2]]
        self.sit_pose = [[0,0,0],[0,0,0],[0,np.pi/3,-np.pi*2/3],[0,np.pi/3,-np.pi*2/3]]
        self.tg = Bezier()
        self.control_fre = 50
        self.t13 =  0.0
        self.t24 = 0 - 0.5


    def run(self):
        self.pybullet_client.resetSimulation()
        self.pybullet_client.setGravity(0, 0, 0)
        self.pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.planeID = self.pybullet_client.loadURDF("plane.urdf")
        # self.planeID = self.pybullet_client.loadURDF("urdf/plane/plane_implicit.urdf")

        self.robot = self.pybullet_client.loadURDF("urdf/spot_old_2.urdf", [0, 0, 0.3],
                                                   useMaximalCoordinates=False,
                                                   flags=p.URDF_USE_IMPLICIT_CYLINDER,
                                                   baseOrientation = self.pybullet_client.getQuaternionFromEuler([0, 0, np.pi]),
                                                   # useFixedBase = 1
                                                   )
        num = p.getNumJoints(self.robot)
        print(f'num=={num}')

        posi_list = []
        color_list = []

        while True:
            if self.counter <200:
                self.pybullet_client.setGravity(0, 0, 0)
                # for i in range(30):
                #     link_info = p.getLinkState(self.robot,i)
                #     print(f'{i}link_info{link_info[0]}')
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

                # time.sleep(self.show_fre)
                self.leg_controller.positions_control2(self.robot,
                                                   self.stand_pose[0],
                                                   self.stand_pose[1],
                                                   self.stand_pose[2],
                                                   self.stand_pose[3],
                                                   )
                self.counter += 1
                pre_position = p.getLinkState(self.robot, 6)[0]
                p.stepSimulation()

            elif self.counter>=200 and self.counter<500:
                posi, ori =  p.getBasePositionAndOrientation(self.robot)
                # print(f"posi======{posi}")
                self.pybullet_client.setGravity(0, 0, -9.8)
                self.leg_controller.positions_control2(self.robot,
                                                   self.stand_pose[0],
                                                   self.stand_pose[1],
                                                   self.stand_pose[2],
                                                   self.stand_pose[3],
                                                   )
                self.counter += 1
                pre_position = p.getLinkState(self.robot, 6)[0]
                p.stepSimulation()

            elif self.counter >=500 and self.counter<=15000:
                self.pybullet_client.setGravity(0, 0,  -9.8)
                cur_position = p.getLinkState(self.robot, 6)[0]

                # -0.0113, -0.045, 0.165 initial coordinate
                p.addUserDebugLine(pre_position, cur_position, lineColorRGB=[1, 0, 0],lineWidth=3,lifeTime = 10)
                pre_position = cur_position

                x1, y1, z1 = self.tg.curve_generator(self.t13)
                x2, y2, z2 = self.tg.curve_generator(self.t24)



                angle13 = self.leg_controller.IK_L_2(x1,  y1, z1)
                angle24= self.leg_controller.IK_L_2(x2, y2, z2)

                time.sleep(self.show_fre)
                self.leg_controller.positions_control2(self.robot,
                                                   angle13,
                                                   angle24,
                                                   angle24,
                                                   angle13,
                                                 )
                self.counter+=1
                self.t13  += 1./self.control_fre
                self.t24  += 1./self.control_fre

                # time.sleep(self.show_fre)
                # p.stepSimulation()
                p.setRealTimeSimulation(self.robot)

            else:
                time.sleep(self.show_fre)
                self.pybullet_client.setGravity(0, 0, -9.8)
                self.leg_controller.positions_control2(self.robot,
                                                   self.sit_pose[0],
                                                   self.sit_pose[1],
                                                   self.sit_pose[2],
                                                   self.sit_pose[3],
                                                   )
                self.counter+=1
                time.sleep(self.show_fre)
                p.stepSimulation()



if __name__ == '__main__':
    spot = Spot()
    spot.run()