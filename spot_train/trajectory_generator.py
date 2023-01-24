# Fachhochschule Aachen
# Name:Bruce_Xing
# Time: 2023/1/17 11:05

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class Bezier:
    def __init__(self,step_length = 0.1, Tswing = 0.5,Tsupport = 0.5 ):
        self.Tswing = Tswing   # unit = s
        self.Tsupport = Tsupport  # unit = s
        self.step_length  = step_length

        # theta1 =-30 theta2=90
        self.initial_x = -0.05
        self.initial_y = 0.045
        self.initial_z = -0.165

        # in fact, here should be forward kinematics

        # transfer to world coordinate
        self.P0 = np.array([self.initial_x,                                    self.initial_y,  0   + self.initial_z])
        self.P1 = np.array([self.initial_x + self.step_length/10    ,          self.initial_y,  0.1 + self.initial_z])
        self.P2 = np.array([self.initial_x + self.step_length * 9/10,          self.initial_y,  0.1 + self.initial_z])
        self.P3 = np.array([self.initial_x + self.step_length,                 self.initial_y,  0   + self.initial_z])



    def curve_generator(self,t):
        t = t % (self.Tswing+self.Tsupport)
        initial_point = [self.initial_x,self.initial_y,self.initial_z]

        point = initial_point
        if t<0:
            point = initial_point
        if t>=0 and t <= self.Tswing:
            t1 =t *2
            point = self.P0*(1-t1)**3 +\
                    3*self.P1*t1* ((1-t1)**2) + \
                    3*self.P2*(t1**2)*(1-t1)+\
                    self.P3*(t1**3)
        if t> self.Tswing and t <=self.Tswing + self.Tsupport:
            point = [self.initial_x+self.step_length - self.step_length/self.Tsupport * (t-self.Tswing),   0.045  ,  -0.165]
        return point



if __name__ == '__main__':
    tg = Bezier()
    t = 0
    x_set = []
    y_set = []
    z_set = []
    fig = plt.figure()
    ax1 = plt.axes(projection = '3d')

    while(True):
        point=tg.curve_generator(t)
        x_set.append(point[0])
        y_set.append(point[1])
        z_set.append(point[2])
        ax1.plot3D(x_set,y_set,z_set,'red')
        ax1.set_xlim(0, 0.2)
        ax1.set_ylim(-.1, 0.1)
        ax1.set_zlim(-.1, 0.1)
        plt.pause(0.1)
        plt.ioff()
        t = t + 0.025

