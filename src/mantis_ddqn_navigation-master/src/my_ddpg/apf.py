#!/usr/bin/env python
#coding=utf-8
import math
from geometry_msgs.msg import Twist
from gazebo_ddpg import Turtlebot3GymEnv
#####################----APF参数设置-----#####################
"""
        :param start_posi_flag: 起点
        :param target_point: 终点
        :param obs: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threashold: 离目标点小于此值即认为到达目标点
"""


class Vector2d():
    """
    2维向量, 支持加减, 支持常量乘法(右乘)
    """
    def __init__(self, x, y):
        self.deltaX = x
        self.deltaY = y
        self.length = -1
        self.direction = [0, 0]
        self.vector2d_share()

    def vector2d_share(self):
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(
                list()):
            deltaX, deltaY = self.deltaX, self.deltaY
            self.deltaX = deltaY[0] - deltaX[0]
            self.deltaY = deltaY[1] - deltaX[1]
            self.length = math.sqrt(self.deltaX**2 + self.deltaY**2) * 1.0
            if self.length > 0:
                self.direction = [
                    self.deltaX / self.length, self.deltaY / self.length
                ]
            else:
                self.direction = None
        else:
            self.length = math.sqrt(self.deltaX**2 + self.deltaY**2) * 1.0
            if self.length > 0:
                self.direction = [
                    self.deltaX / self.length, self.deltaY / self.length
                ]
            else:
                self.direction = None

    def __add__(self, other):
        """
        + 重载
        :param other:
        :return:
        """
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.vector2d_share()
        return vec

    def __sub__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.vector2d_share()
        return vec

    def __mul__(self, other):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.deltaX *= other
        vec.deltaY *= other
        vec.vector2d_share()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def reverse(self):
        vec = Vector2d(self.deltaX, self.deltaY)
        vec.vector2d_share()
        vec.direction[0] = -vec.direction[0]
        vec.direction[1] = -vec.direction[1]
        return vec

    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, length:{}, direction:{}'.format(
            self.deltaX, self.deltaY, self.length, self.direction)


#####################----APF参数设置-----#####################
"""
        :param start_posi_flag: 起点
        :param target_point: 终点
        :param obs: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threashold: 离目标点小于此值即认为到达目标点
"""


class APF():
    def __init__(self, env, max_action):
        self.env = env
        self.current_pos = Vector2d(0, 0)
        self.goal = Vector2d(0, 0)
        self.obstacles = [Vector2d(0, 0)]
        self.k_att = 5
        self.k_rep = 2
        self.rr = 2  # 斥力作用范围
        self.max_speed = max_action[0]
        self.max_angular = max_action[1]
        self.min_att_dist = 2

    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        goal_to_current_pos = self.goal - self.current_pos
        att = goal_to_current_pos * self.k_att  # 方向由机器人指向目标点
        att_angular = math.atan2(att.deltaY, att.deltaX)
        return att_angular

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        self.obstacles = [Vector2d(self.env.obs[0], self.env.obs[1])]
        for obstacle in self.obstacles:
            if obstacle.deltaX == 0 and obstacle.deltaY == 0:
                rep_angular = 0
                break
            else:
                obs_to_rob = obstacle
                if obs_to_rob.length > self.rr:
                    rep_angular = 0
                else:

                    rep_angular = math.atan2(obs_to_rob.deltaY,
                                             obs_to_rob.deltaX)
                    if rep_angular > 0:
                        rep_angular = rep_angular - math.pi
                    else:
                        rep_angular = rep_angular + math.pi
                    # 让机器人距离障碍物越近，拐弯角度的比例越大（在基础的对角的角度基础作为最大）
                    rep_angular = -rep_angular / self.rr * obs_to_rob.length + rep_angular

        return rep_angular

    def out_put_velocity(self):
        # 初始化数据
        self.env.getOdomData()
        self.current_pos = Vector2d(self.env.position_x, self.env.position_y)
        self.goal = Vector2d(self.env.targetPointX, self.env.targetPointY)

        att_angular = self.attractive()
        rep_angular = self.repulsion()
        ######----角度计算-----##########
        # 注意这里必须是atan2，否则角度会出问题
        # print("att_angular: ", att_angular / math.pi * 180, "rep_angular: ",
        #       rep_angular / math.pi * 180)
        angular = att_angular + rep_angular - self.env.angPos
        print("angular: ", angular / math.pi * 180)
        if angular > math.pi:
            angular = math.pi - 2 * math.pi
        if angular < -math.pi:
            angular = math.pi + 2 * math.pi
        angular = angular / 3
        speed_x = (self.current_pos - self.goal).length
        vel_msg = Vector2d(speed_x, angular)
        return vel_msg

    def apf_robot(self):
        motion_param = self.out_put_velocity()
        vel_msg = Twist()
        vel_msg.linear.x = motion_param.deltaX
        vel_msg.angular.z = motion_param.deltaY
        # 速度和角速度限制
        if vel_msg.linear.x > self.max_speed:
            vel_msg.linear.x = self.max_speed
        if vel_msg.angular.z > self.max_angular:
            vel_msg.angular.z = self.max_angular
        if vel_msg.angular.z < -self.max_angular:
            vel_msg.angular.z = -self.max_angular
        # 令机器人转向的的时候线速度为0
        # if abs(vel_msg.angular.z) > 10.0 / 180 * math.pi:
        #     vel_msg.linear.x = 0
        action = [vel_msg.linear.x, vel_msg.angular.z]
        return action
