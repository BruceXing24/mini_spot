# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/2/13 11:59


# this file is for CPG refer to paper
import numpy as np
import time
import matplotlib.pyplot as plt
import spot_leg


class CPG:
    def __init__(self, step_length=0.05, ground_clearance=0.05, ground_penetration=0.01, Tstance=0.6, Tswing=0.4):
        self.step_length = step_length
        self.zClearSwing = ground_clearance
        self.zClearStance = ground_penetration
        self.Tstance = Tstance
        self.Tswing = Tswing

        # self.Tswing_walk = 0.25
        # self.Tstance_walk = 0.75

        self.trot_timer = 0.0
        self.pacing_timer = 0.0

        self.Tperiod = self.Tstance + self.Tswing

        self.spot_width = 0.15  # robot size
        self.spot_length = 0.3
        # self.turn_radius = 0.3
        self.spot_leg = spot_leg.Leg()

        # local refer to hip
        self.init_x = -0.01  # 0.019
        self.init_y = 0.055
        self.init_z = -0.173

        self.leg_initalxyz = [-0.01, 0.055, -0.173]

        self.body_width = 0.093
        self.body_length = 0.188

        self.initl_xyz = np.array([[self.init_x + self.body_length / 2, self.init_y + self.body_width / 2, -0.173],
                                   [self.init_x + self.body_length / 2, self.init_y - self.body_width / 2, -0.173],
                                   [self.init_x - self.body_length / 2, self.init_y + self.body_width / 2, -0.173],
                                   [self.init_x - self.body_length / 2, self.init_y - self.body_width / 2, -0.173]
                                   ])

    def curve_generator(self, t):
        t = t % self.Tperiod
        if t >= 0 and t < self.Tswing:
            pha = t * np.pi / self.Tswing
        else:
            pha = t * np.pi / self.Tstance + (np.pi - self.Tswing / self.Tstance * np.pi)
        x_des = -self.step_length * np.cos(pha)

        # print(f"t=={t}")
        # print(f"x_des=={x_des}")

        if np.sin(pha) >= 0:
            z = self.zClearSwing
        else:
            z = self.zClearStance
        z_des = z * np.sin(pha)
        return x_des, z_des

    def foot_trajectory(self, t, fxyz, direction, leg_name, turn_radius):
        x_t, z_t = self.curve_generator(t)
        x_r = x_t
        y_r = 0
        z_r = z_t
        refer_xyz = np.array([x_r, y_r, z_r])

        if direction == 'straight':
            return refer_xyz + fxyz
        elif direction == 'right':
            alpha, beta = self.get_angle(turn_radius)
            if leg_name == 'FL':
                return self.get_Rmatrix(-beta).dot(refer_xyz) + fxyz
            if leg_name == 'FR':
                return self.get_Rmatrix(-alpha).dot(refer_xyz) + fxyz
            if leg_name == 'BL':
                return self.get_Rmatrix(beta).dot(refer_xyz) + fxyz
            if leg_name == 'BR':
                return self.get_Rmatrix(alpha).dot(refer_xyz) + fxyz
        elif direction == 'left':
            alpha, beta = self.get_angle(turn_radius)
            if leg_name == 'FL':
                return self.get_Rmatrix(alpha).dot(refer_xyz) + fxyz
            if leg_name == 'FR':
                return self.get_Rmatrix(beta).dot(refer_xyz) + fxyz
            if leg_name == 'BL':
                return self.get_Rmatrix(-alpha).dot(refer_xyz) + fxyz
            if leg_name == 'BR':
                return self.get_Rmatrix(-beta).dot(refer_xyz) + fxyz
            print(f"alpha=={alpha},beta==={beta}")

    def get_angle(self, turn_radius):
        alpha = np.arctan(self.spot_length / (2 * turn_radius - self.spot_width))
        beta = np.arctan(self.spot_length / (2 * turn_radius + self.spot_width))
        return np.abs(alpha), np.abs(beta)

    def get_Rmatrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), -np.cos(theta), 0],
                         [0, 0, 1]])

    def trot_generator(self, t, direction, turn_radius):
        self.Tstance = 0.3
        self.Tswing = 0.3
        self.Tperiod = self.Tstance + self.Tswing
        # trot gait should be Tswing:Tstance 1:1

        Txyz_FL = self.foot_trajectory(t, self.leg_initalxyz, direction, 'FL',
                                       turn_radius)
        Txyz_FR = self.foot_trajectory(t + self.Tswing, self.leg_initalxyz, direction,
                                       'FR', turn_radius)
        Txyz_BL = self.foot_trajectory(t + self.Tswing, self.leg_initalxyz, direction,
                                       'BL', turn_radius)
        Txyz_BR = self.foot_trajectory(t, self.leg_initalxyz, direction, 'BR',
                                       turn_radius)
        FL_1, FL_2, FL_3 = self.spot_leg.IK_L_2(Txyz_FL[0], Txyz_FL[1], Txyz_FL[2])
        FR_1, FR_2, FR_3 = self.spot_leg.IK_L_2(Txyz_FR[0], Txyz_FR[1], Txyz_FR[2])
        BL_1, BL_2, BL_3 = self.spot_leg.IK_L_2(Txyz_BL[0], Txyz_BL[1], Txyz_BL[2])
        BR_1, BR_2, BR_3 = self.spot_leg.IK_L_2(Txyz_BR[0], Txyz_BR[1], Txyz_BR[2])
        return np.array([FL_1, FL_2, FL_3, FR_1, FR_2, FR_3, BL_1, BL_2, BL_3, BR_1, BR_2, BR_3]) * 180 / np.pi

    def walk_generator(self, t, direction, turn_radius):
        # walk gait should be Tswing:Tstance 1:3
        
        self.Tstance = 0.75*4
        self.Tswing = 0.25*4

        self.Tperiod = self.Tstance + self.Tswing

        Txyz_FL = self.foot_trajectory(t, self.leg_initalxyz, direction, 'FL',
                                       turn_radius)
        Txyz_FR = self.foot_trajectory(t + self.Tswing, self.leg_initalxyz, direction,
                                       'FR', turn_radius)
        Txyz_BL = self.foot_trajectory(t + 2 * self.Tswing, self.leg_initalxyz, direction,
                                       'BL', turn_radius)
        Txyz_BR = self.foot_trajectory(t + 3 * self.Tswing, self.leg_initalxyz, direction, 'BR',
                                       turn_radius)
        FL_1, FL_2, FL_3 = self.spot_leg.IK_L_2(Txyz_FL[0], Txyz_FL[1], Txyz_FL[2])
        FR_1, FR_2, FR_3 = self.spot_leg.IK_L_2(Txyz_FR[0], Txyz_FR[1], Txyz_FR[2])
        BL_1, BL_2, BL_3 = self.spot_leg.IK_L_2(Txyz_BL[0], Txyz_BL[1], Txyz_BL[2])
        BR_1, BR_2, BR_3 = self.spot_leg.IK_L_2(Txyz_BR[0], Txyz_BR[1], Txyz_BR[2])
        return np.array([FL_1, FL_2, FL_3, FR_1, FR_2, FR_3, BL_1, BL_2, BL_3, BR_1, BR_2, BR_3]) * 180 / np.pi


    def gallop_generator(self):
        pass

    def reset_time(self):
        self.trot_timer = 0.0


if __name__ == '__main__':
    CPG_controller = CPG(ground_penetration=0.02)
    t = 0
    x_set = [[], [], [], []]
    y_set = [[], [], [], []]
    z_set = [[], [], [], []]
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')

    while True:
        t = 0
        for i in range(100):
            t += 1. / 100
            # x, z = CPG_controller.curve_generator(t)
            # x_set.append(x)
            # y_set.append(0)
            # z_set.append(z)
            # ax1.plot3D(x_set, y_set, z_set, 'red')

            Txyz_FL = CPG_controller.foot_trajectory(t, CPG_controller.leg_initalxyz, 'straight', 'FL', 0.8)
            Txyz_FR = CPG_controller.foot_trajectory(t, CPG_controller.initl_xyz[1], 'right', 'FR', 0.8)
            Txyz_BL = CPG_controller.foot_trajectory(t, CPG_controller.initl_xyz[2], 'right', 'BL', 0.8)
            Txyz_BR = CPG_controller.foot_trajectory(t, CPG_controller.initl_xyz[3], 'right', 'BR', 0.8)

            print(f'Txyz_FL==={Txyz_FL}')

            x_set[0].append(Txyz_FL[0])
            y_set[0].append(Txyz_FL[1])
            z_set[0].append(Txyz_FL[2])
            x_set[1].append(Txyz_FR[0])
            y_set[1].append(Txyz_FR[1])
            z_set[1].append(Txyz_FR[2])
            x_set[2].append(Txyz_BL[0])
            y_set[2].append(Txyz_BL[1])
            z_set[2].append(Txyz_BL[2])
            x_set[3].append(Txyz_BR[0])
            y_set[3].append(Txyz_BR[1])
            z_set[3].append(Txyz_BR[2])

            ax1.plot3D(x_set[0], y_set[0], z_set[0], 'blue')
            ax1.plot3D(x_set[1], y_set[1], z_set[1], 'black')
            ax1.plot3D(x_set[2], y_set[2], z_set[2], 'red')
            ax1.plot3D(x_set[3], y_set[3], z_set[3], 'red')

            ax1.set_xlim(-.15, 0.15)
            ax1.set_ylim(-.15, 0.15)
            ax1.set_zlim(-.15, 0.15)
            plt.pause(0.1)
            plt.ioff()
