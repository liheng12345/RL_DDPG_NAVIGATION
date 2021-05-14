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


class APF(Turtlebot3GymEnv):
    def __init__(self, Turtlebot3GymEnv):
        super(APF, self).__init__()
        self.current_pos = Vector2d(0, 0)
        self.goal = Vector2d(0, 0)
        self.obstacles = [Vector2d(0, 0)]
        self.k_att = 8
        self.k_rep = 1
        self.rr = 0.9  # 斥力作用范围
        self.max_speed = 0.3
        self.max_angular = 0.8

    def attractive(self):
        """
        引力计算
        :return: 引力
        """
        att = (self.goal - self.current_pos) * self.k_att  # 方向由机器人指向目标点
        return att

    def repulsion(self):
        """
        斥力计算, 改进斥力函数, 解决不可达问题
        :return: 斥力大小
        """
        rep = Vector2d(0, 0)  # 所有障碍物总斥力
        for obstacle in self.obstacles:
            if obstacle.deltaX == 0 and obstacle.deltaY == 0:
                break
            else:
                # obstacle = Vector2d(0, 0)
                obs_to_rob = self.current_pos - obstacle
                rob_to_goal = self.goal - self.current_pos
                if (obs_to_rob.length > self.rr):  # 超出障碍物斥力影响范围
                    pass
                else:
                    rep_1 = Vector2d(
                        obs_to_rob.direction[0], obs_to_rob.direction[1]) * (
                            self.k_rep *
                            (1.0 / obs_to_rob.length - 1.0 / self.rr) /
                            (obs_to_rob.length**2) * (rob_to_goal.length**2))
                    rep_2 = Vector2d(
                        rob_to_goal.direction[0],
                        rob_to_goal.direction[1]) * (self.k_rep * (
                            (1.0 / obs_to_rob.length - 1.0 / self.rr)**2) *
                                                     rob_to_goal.length)
                    rep += (rep_1 + rep_2)
        return rep

    def out_put_velocity(self):
        # 初始化数据
        self.getOdomData()
        self.current_pos = Vector2d(self.position_x, self.position_y)
        self.goal = Vector2d(self.targetPointX, self.targetPointY)

        f_vec = self.attractive() + self.repulsion()
        ######----角度计算-----##########
        # 注意这里必须是atan2，否则角度会出问题
        angular = math.atan2(f_vec.direction[1],
                             f_vec.direction[0]) - self.angPos
        if angular > math.pi:
            angular = math.pi - 2 * math.pi
        if angular < -math.pi:
            angular = math.pi + 2 * math.pi
        speed_x = (self.current_pos - self.goal).length
        vel_msg = Vector2d(speed_x, angular)
        return vel_msg

    def apf_robot(self):
        motion_param = self.out_put_velocity()
        vel_msg = Twist()
        vel_msg.linear.x = motion_param.direction[0]
        vel_msg.angular.z = motion_param.direction[1]
        # 速度和角速度限制
        if vel_msg.linear.x > self.max_speed:
            vel_msg.linear.x = self.max_speed
        if vel_msg.angular.z > self.max_angular:
            vel_msg.angular.z = self.max_angular
        # 令机器人转向的的时候线速度为0
        # if abs(vel_msg.angular.z) > 10.0 / 180 * math.pi:
        #     vel_msg.linear.x = 0
        action = [vel_msg.linear.x, vel_msg.angular.z]
        return action
