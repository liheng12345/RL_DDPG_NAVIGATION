#!/usr/bin/env python
#coding=utf-8
import rospy
import roslib
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Illuminance
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ContactState
from tf.transformations import euler_from_quaternion, quaternion_from_euler

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
#记录机器人的实时位置
start_posi_flag = (0, 0)
#设定的目标点
target_point = [[1, -1.5], [1.5, -4.5]]
#障碍物的位置，通过gazebo观察，前三个是桶形障碍物，后面两个是车库的
obstacles = [[0.41, -2.58], [2.41, -2.58], [3.41, -2.58], [1, -1.1],
             [1.5, -1.5]]
max_speed = 0.05
max_angular = 0.8
goal_threashold = 0.2
k_att, k_rep = 8, 1.0
#距离障碍物多远开始避障
rr = 0.9

preError = 0.00
linearVel = 0.025
angularVel = math.pi / 20.00
angularVel_D = math.pi / 13.33
desPosX = 0.00
desPosY = 0.00
desAngPos = 0.00
roundN = 0
posX = 0.00
posY = 0.00
angPos = 0.00
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


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


class APF_Improved():
    def __init__(self, start, goal, obstacles, k_att, k_rep, rr,
                 goal_threshold, max_speed, max_angular, angPos):
        self.start = Vector2d(start[0], start[1])
        self.current_pos = Vector2d(start[0], start[1])
        self.goal = Vector2d(goal[0], goal[1])
        self.obstacles = [Vector2d(OB[0], OB[1]) for OB in obstacles]
        self.k_att = k_att
        self.k_rep = k_rep
        self.rr = rr  # 斥力作用范围
        self.iters = 0
        self.goal_threashold = goal_threshold
        self.path = list()
        self.is_path_plan_success = False
        self.delta_t = 0.01
        self.max_speed = max_speed
        self.max_angular = max_angular
        self.angPos = angPos

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
                rep_2 = Vector2d(rob_to_goal.direction[0],
                                 rob_to_goal.direction[1]) * (self.k_rep * (
                                     (1.0 / obs_to_rob.length - 1.0 / self.rr)
                                     **2) * rob_to_goal.length)
                rep += (rep_1 + rep_2)
        return rep

    def out_put_velocity(self):
        f_vec = self.attractive() + self.repulsion()
        ######----角度计算-----##########
        #注意这里必须是atan2，否则角度会出问题
        angular = math.atan2(f_vec.direction[1],
                             f_vec.direction[0]) - self.angPos
        if angular > math.pi:
            angular = math.pi - 2 * math.pi

        # speed_x=math.sqrt(f_vec.direction[0]**2+f_vec.direction[0]**2)
        speed_x = (self.current_pos - self.goal).length

        # speed_x= f_vec.direction[0]/math.sqrt(f_vec.direction[0]**2+f_vec.direction[1]**2)*self.max_speed
        # speed_y=f_vec.direction[1]/math.sqrt(f_vec.direction[0]**2+f_vec.direction[1]**2)*self.max_angular
        vel_msg = Vector2d(speed_x, angular)
        # print("speed_x:",speed_x,angular)
        return vel_msg


# timeNow = rospy.get_rostime().to_sec()
def callBack_Odom(msg):
    global angPos
    global posX
    global posY
    global start_posi_flag
    odomPose = msg.pose.pose
    orientation_list = [
        odomPose.orientation.x, odomPose.orientation.y, odomPose.orientation.z,
        odomPose.orientation.w
    ]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    angPos = yaw
    posX = odomPose.position.x
    posY = odomPose.position.y
    #获取机器人开机的位置
    start_posi_flag = [posX, posY]


def apf_robot(intialpoint, endpoint, obstacles):
    global cmd_vel_pub, goal_threashold, k_att, k_rep, rr, angPos, max_speed, max_angular
    """
        :param start_posi_flag: 起点
        :param target_point: 终点
        :param obs: 障碍物列表，每个元素为Vector2d对象
        :param k_att: 引力系数
        :param k_rep: 斥力系数
        :param rr: 斥力作用范围
        :param goal_threashold: 离目标点小于此值即认为到达目标点
        :param angPos: 机器人当前朝向
    """
    apf = APF_Improved(intialpoint, endpoint, obstacles, k_att, k_rep, rr,
                       goal_threashold, max_speed, max_angular, angPos)
    motion_param = apf.out_put_velocity()

    vel_msg = Twist()
    vel_msg.linear.x = 0.1 * motion_param.direction[0]
    vel_msg.angular.z = 0.8 * motion_param.direction[1]
    # 速度和角速度限制
    if vel_msg.linear.x > max_speed:
        vel_msg.linear.x = max_speed
    if vel_msg.angular.z > max_angular:
        vel_msg.angular.z = max_angular
    # 令机器人转向的的时候线速度为0
    if abs(vel_msg.angular.z) > 10.0 / 180 * math.pi:
        vel_msg.linear.x = 0
    cmd_vel_pub.publish(vel_msg)


def test_apf():
    rate = rospy.Rate(10)
    count = 0
    for goal_point in target_point:
        count = count + 1
        while (math.sqrt((posX - goal_point[0])**2 +
                         (posY - goal_point[1])**2)) > goal_threashold:
            if rospy.is_shutdown():
                break
            apf_robot(start_posi_flag, goal_point, obstacles)
            rate.sleep()
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        print("到达第%d个目标点" % (count))

        #  这里为止已经到了第一个目标点，后面你先加入你的程序，就在后面实现即可
        #  ---------你的程序------------ #
        #  ---------------------------- #
        #  ---------------------------- #
        #  ---------------------------- #
        #  ---------------------------- #
        #  ---------------------------- #
        #  ---------------------------- #
        #  你的程序结束以后，就会自动进行下一个目标点的运动


def callBack_Bumper(msg):
    global bumperVel
    bumperVel = msg.states


def move_forward1():
    global roundN, desPosX, linearVel, cmd_vel_pub, preError
    vel_msg = Twist()
    vel_msg.linear.x = 0.4
    cmd_vel_pub.publish(vel_msg)
    print("sdfsaf")


def move_forward():
    global roundN, desPosX, linearVel, cmd_vel_pub, preError

    if roundN == 0:
        rospy.sleep(2.)

    vel_msg = Twist()

    if roundN < 10:
        if bumperVel == []:
            desPosX = posX + 0.2
            vel_msg.linear.x = linearVel
            cmd_vel_pub.publish(vel_msg)
            rospy.sleep(4.)
        else:
            vel_msg.linear.x = -0.025
            cmd_vel_pub.publish(vel_msg)
            rospy.sleep(4.)
            roundN += 1


def move_backward():
    global roundN, desPosX, linearVel, cmd_vel_pub, preError

    if roundN == 0:
        rospy.sleep(2.)

    vel_msg = Twist()

    if roundN < 10:
        desPosX = posX - 0.2
        vel_msg.linear.x = -linearVel
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(4.)

        vel_msg.linear.x = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(4.)

        error = desPosX - posX + preError
        print(str(roundN) + ": " + str(error))
        roundN += 1


def move_square():
    global roundN, desPosX, desPosY, desAngPos, linearVel, angularVel, cmd_vel_pub

    if roundN == 0:
        rospy.sleep(2.)

        desPosX = posX
        desPosY = posY
        desAngPos = angPos

    vel_msg = Twist()

    if roundN < 4:
        step = 0
        while step < 4:
            vel_msg.linear.x = linearVel
            vel_msg.angular.z = 0
            cmd_vel_pub.publish(vel_msg)
            rospy.sleep(10.)
            vel_msg.linear.x = 0
            vel_msg.angular.z = angularVel
            cmd_vel_pub.publish(vel_msg)
            rospy.sleep(5.)
            step += 1

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(2.)

        error = math.sqrt((desPosX - posX)**2 + (desPosY - posY)**2)
        error_angle = desAngPos - angPos
        print(
            str(roundN) + ": " + "xy: " + str(error) + "  angle: " +
            str(error_angle))
        roundN += 1


def move_diagonal():
    global roundN, desPosX, desPosY, desAngPos, linearVel, angularVel, angularVel_D, cmd_vel_pub

    if roundN == 0:
        rospy.sleep(2.)
        desPosX = posX + 0.8 + 0.25 * math.sqrt(2)
        desPosY = posY - (0.5 - 0.25 * math.sqrt(2))
        desAngPos = angPos - math.pi / 4

    if roundN == 1:
        arc = math.sqrt((0.8 + 0.25 * math.sqrt(2))**2 +
                        (0.5 - 0.25 * math.sqrt(2))**2)
        ang = math.atan((0.5 - 0.25 * math.sqrt(2)) /
                        (0.8 + 0.25 * math.sqrt(2))) + math.pi / 4
        desPosX = desPosX + arc * math.sin(ang)
        desPosY = desPosY + arc * math.cos(ang)
        desAngPos = desAngPos - math.pi / 4

    if roundN == 2:
        desPosX = desPosX + (0.5 - 0.25 * math.sqrt(2))
        desPosY = desPosY + 0.8 + 0.25 * math.sqrt(2)
        desAngPos = desAngPos - math.pi / 4

    vel_msg = Twist()

    if roundN < 3:
        vel_msg.linear.x = linearVel
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(16.)

        vel_msg.linear.x = 0
        vel_msg.angular.z = angularVel
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(5.)

        vel_msg.linear.x = linearVel
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(10.)

        vel_msg.linear.x = 0
        vel_msg.angular.z = -angularVel_D
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(5.)

        vel_msg.linear.x = linearVel
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(10.)

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        cmd_vel_pub.publish(vel_msg)
        rospy.sleep(2.)

        error = math.sqrt((desPosX - posX)**2 + (desPosY - posY)**2)
        error_angle = desAngPos - angPos
        print(
            str(roundN) + ": " + "xy: " + str(error) + "  angle: " +
            str(error_angle))
        roundN += 1


def turtuleBotGo():
    rospy.init_node('turtleBotGo', anonymous=True)

    rospy.Subscriber('/odom', Odometry, callBack_Odom)

    rospy.Subscriber('/robot/bumper_states', ContactsState, callBack_Bumper)

    while not rospy.is_shutdown():
        test_apf()
        break


if __name__ == '__main__':
    try:
        turtuleBotGo()
    except rospy.ROSInterruptException:
        pass