#-*- coding:utf-8 –*-
import rospy
import time
import numpy as np
import math

import random

from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
"""
地图选择
"""
SELECT_MAP = "maze1"


class AgentPosController():
    '''
    随机传送机器人位置，在世界复位的时候
    '''
    def __init__(self):
        self.agent_model_name = "turtlebot3_waffle"

    def teleportfixed(self):
        '''
        传送机器人位置
        返回[posX,posY]
        '''
        model_state_msg = ModelState()
        model_state_msg.model_name = self.agent_model_name

        # 选择机器人的位置
        pose = Pose()
        pose.position.x, pose.position.y = [-1, 0]

        model_state_msg.pose = pose
        model_state_msg.twist = Twist()

        model_state_msg.reference_frame = "world"

        # 开始机器人传送
        isTeleportSuccess = False

        if not isTeleportSuccess:
            try:
                rospy.wait_for_service('/gazebo/set_model_state')
                telep_model_prox = rospy.ServiceProxy(
                    '/gazebo/set_model_state', SetModelState)
                telep_model_prox(model_state_msg)
                isTeleportSuccess = True
            except Exception as e:
                rospy.logfatal("设置机器人位置成功 " + str(e))

        return pose.position.x, pose.position.y

    def teleportRandom(self):
        '''
        传送机器人位置
        返回[posX,posY]
        '''

        model_state_msg = ModelState()
        model_state_msg.model_name = self.agent_model_name

        if SELECT_MAP == "maze1":
            # 随机选择机器人的位置

            xy_list = [[-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
                       [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
                       [5.5, -1.5], [5.5, -0.5], [5.5, 0.5], [5.5, 1.5]]
            # pose = Pose()
            # pose.position.x = round(random.uniform(-2, 5), 3)
            # pose.position.y = round(random.uniform(-1.5, 1.5), 3)

        # 随机选择机器人的位置
        pose = Pose()
        pose.position.x, pose.position.y = random.choice(xy_list)
        model_state_msg.pose = pose
        model_state_msg.twist = Twist()

        model_state_msg.reference_frame = "world"

        # 开始机器人传送
        isTeleportSuccess = False

        if not isTeleportSuccess:
            try:
                rospy.wait_for_service('/gazebo/set_model_state')
                telep_model_prox = rospy.ServiceProxy(
                    '/gazebo/set_model_state', SetModelState)
                telep_model_prox(model_state_msg)
                isTeleportSuccess = True
            except Exception as e:
                rospy.logfatal("设置机器人位置成功 " + str(e))

        return pose.position.x, pose.position.y


class GoalController():
    """
    机器人目标控制
    """
    def __init__(self):
        self.model_path = "../../models/gazebo/goal_sign/model.sdf"
        f = open(self.model_path, 'r')
        self.model = f.read()

        self.goal_position = Pose()
        self.goal_position.position.x = None  # Initial positions
        self.goal_position.position.y = None
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        self.model_name = 'goal_sign'
        self.check_model = False  # 检查是否有模型

    def respawnModel(self):
        '''
        在gazebo中生成地面上的圈
        '''
        self.calculate_fix_goal()
        if not self.check_model:
            try:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model',
                                                      SpawnModel)
                spawn_model_prox(self.model_name, self.model,
                                 'robotos_name_space', self.goal_position,
                                 "world")
                self.check_model = True
            except Exception as e:
                rospy.logfatal("生成目标点失败 " + str(e))

    def deleteModel(self):
        '''
        在gazebo中删除圈
        '''
        if self.check_model:
            try:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model',
                                                    DeleteModel)
                del_model_prox(self.model_name)
                self.check_model = False
            except Exception as e:
                rospy.logfatal("Error when deleting the goal sign " + str(e))

    def calculate_fix_goal(self):
        """
        随机返回一个目标点在gazebo中
        """
        # Wait for deleting
        time.sleep(0.5)

        # self.goal_position.position.x, self.goal_position.position.y = [2, 0.5]
        self.goal_position.position.x, self.goal_position.position.y = [5, 0]
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y

    def getTargetPoint(self):
        return self.goal_position.position.x, self.goal_position.position.y


class Turtlebot3GymEnv():
    '''
    gazebo的环境配置
    '''
    def __init__(self):
        # Initialize the node
        rospy.init_node('turtlebot3_gym_env', anonymous=True)
        # 连接gazebo
        self.velPub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation',
                                              Empty)

        self.laserPointCount = 24  # 总共设置24个雷达点，需要在urdf中进行设置
        self.minCrashRange = 0.2  # 碰撞阈值
        self.laserMinRange = 0.2  # 雷达检测范围
        self.laserMaxRange = 10.0
        self.stateSize = self.laserPointCount + 4  # 机器人的状态个数【Laser(arr), heading, distance, obstacleMinRange, obstacleAngle】
        self.actionSize = 1  # 机器人动作个数
        self.targetDistance = 0  # 离目标点的距离

        # 定义目标点生成器
        self.goalCont = GoalController()
        # 定义机器人控制器
        self.agentController = AgentPosController()
        # 获取目标点的位置
        self.targetPointX, self.targetPointY = self.goalCont.calculate_fix_goal(
        )
        self.past_distanceToTarget = 0
        self.past_AngleToTarget = 0
        self.angPos = 0
        self.position_x = 0
        self.position_y = 0
        # 障碍物信息
        self.obs = [0, 0]
        # 默认意味机器人没有到达了目标点，再开始重新复位环境
        self.isTargetReached = True
        self.getOdomData()
        self.adaptive_cmd_flag = False

    def pauseGazebo(self):
        '''
        暂停仿真
        '''
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except Exception:
            print("gazebo暂停仿真失败")

    def unpauseGazebo(self):
        '''
        仿真开始
        '''
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except Exception:
            print("gazebo仿真开始失败")

    def resetGazebo(self):
        '''
        复位gazebo环境
        '''
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except Exception:
            print("复位环境失败")

    def getLaserData(self):
        '''
        获取雷达数据
        '''
        # self.unpauseGazebo()
        try:
            laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            return laserData
        except Exception as e:
            rospy.logfatal("获取雷达数据失败 " + str(e))

    def getOdomData(self):
        '''
        返回机器人的位置和朝向
        '''
        # self.unpauseGazebo()
        try:
            odomData = rospy.wait_for_message('/odom', Odometry, timeout=5)
            odomData = odomData.pose.pose
            quat = odomData.orientation
            quatTuple = (
                quat.x,
                quat.y,
                quat.z,
                quat.w,
            )

            r = math.atan2(
                2 *
                (quatTuple[3] * quatTuple[0] + quatTuple[1] * quatTuple[2]),
                1 - 2 *
                (quatTuple[0] * quatTuple[0] + quatTuple[1] * quatTuple[1]))
            p = math.asin(
                2 *
                (quatTuple[3] * quatTuple[1] - quatTuple[2] * quatTuple[0]))
            y = math.atan2(
                2 *
                (quatTuple[3] * quatTuple[2] + quatTuple[0] * quatTuple[1]),
                1 - 2 *
                (quatTuple[2] * quatTuple[2] + quatTuple[1] * quatTuple[1]))
            roll = r * 180 / math.pi
            pitch = p * 180 / math.pi
            yaw = y * 180 / math.pi
            robotX = odomData.position.x
            robotY = odomData.position.y
            self.angPos = y
            self.position_x = robotX
            self.position_y = robotY
            return yaw, robotX, robotY

        except Exception as e:
            rospy.logfatal("获取odom失败 " + str(e))

    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX,
                         robotY):
        '''
        计算机器人和目标点的朝向
        '''
        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle * 180 / math.pi - yaw
        if heading > 180:
            heading -= 2 * 180

        elif heading < -180:
            heading += 2 * 180

        return round(heading, 2)

    def calcDistance(self, x1, y1, x2, y2):
        '''
        返回两个点的距离
        '''
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def calculateState(self, laserData, odomData):
        '''
        计算机器人目前的状态
        返回 
        【laserData + [
            heading, distance, obstacleMinRange, obstacleAngle
        ], isCrash  
        总共是360+4+1
        '''

        heading = self.calcHeadingAngle(self.targetPointX, self.targetPointY,
                                        *odomData)
        _, robotX, robotY = odomData
        distance = self.calcDistance(robotX, robotY, self.targetPointX,
                                     self.targetPointY)

        isCrash = False  # If robot hit to an obstacle
        laserData = list(laserData.ranges)
        for i in range(len(laserData)):
            # 判断是否发生碰撞
            if (self.minCrashRange > laserData[i] > 0):
                isCrash = True
            # 判断是否应该强制调整机器人的位置
            if (self.minCrashRange + 0.6 > laserData[i] > 0):
                self.adaptive_cmd_flag = True
            # 雷达数据无穷大的时候强制设置为10
            if np.isinf(laserData[i]):
                laserData[i] = self.laserMaxRange
            if np.isnan(laserData[i]):
                laserData[i] = 0
        # 保留两位小数，四舍五入
        obstacleMinRange = round(min(laserData), 2)
        # 找到最小的雷达数据的索引
        obstacleAngle = np.argmin(laserData)
        angle = obstacleAngle / len(laserData) * 360
        if angle > 180:
            angle = angle - 2 * 180
        if angle > 90 and angle <= 180:
            x = -obstacleMinRange * math.cos(math.pi - angle / 180 * math.pi)
            y = obstacleMinRange * math.sin(math.pi - angle / 180 * math.pi)
        elif angle >= -180 and angle < -90:
            x = -obstacleMinRange * math.cos(math.pi + angle / 180 * math.pi)
            y = -obstacleMinRange * math.sin(math.pi + angle / 180 * math.pi)
        elif angle >= -90 and angle < 0:
            x = obstacleMinRange * math.cos(-angle / 180 * math.pi)
            y = -obstacleMinRange * math.sin(-angle / 180 * math.pi)
        else:
            x = obstacleMinRange * math.cos(angle / 180 * math.pi)
            y = obstacleMinRange * math.sin(angle / 180 * math.pi)
        self.obs = [x, y]
        # 计算到目标点的距离
        if obstacleMinRange < 0.2:
            isCrash = True

        return laserData + [
            heading, distance, obstacleMinRange, obstacleAngle
        ], isCrash

    def step(self, action):
        '''
        在仿真环境中执行一步，并且暂停仿真，返回机器人状态信息
        '''
        # gazebo开启
        # self.unpauseGazebo()
        reward = 0

        # 获取观测
        laserData = self.getLaserData()
        odomData = self.getOdomData()

        # gazebo仿真暂停
        # self.pauseGazebo()

        # 获取目标点的位置
        self.targetPointX, self.targetPointY = self.goalCont.getTargetPoint()
        if odomData is not None:
            state, isCrash = self.calculateState(laserData, odomData)
        # 计算到目标点的距离
        current_distance = state[-3]
        # 计算到目标点的朝向
        AngleToTarget = state[-4]
        done = False

        # 计算机器人朝向，速度
        velCmd = Twist()

        # 强制调整机器人位置
        # if self.adaptive_cmd_flag:
        #     velCmd.angular.z = 0.5 * AngleToTarget / 180 * math.pi
        #     print("force turn")
        #     if abs(AngleToTarget) < 10:
        #         velCmd.linear.x = action[0]
        #     else:
        #         velCmd.linear.x = 0
        # else:
        #     # 计算机器人朝向，速度
        #     velCmd.linear.x = action[0] / 2
        #     velCmd.angular.z = action[1]
        # self.adaptive_cmd_flag = False

        # 输出机器人朝向，速度
        velCmd.linear.x = action[0]
        velCmd.angular.z = action[1]
        self.velPub.publish(velCmd)

        if isCrash:
            done = True
            reward = -2
        elif current_distance < 0.2:  # Reached to target
            self.isTargetReached = True
            reward = 4
            done = True
        else:
            # 没有到达目标点也没有发生碰撞,计算奖赏,距离越近，奖赏越大
            # distance_rate = 100 * (self.past_distanceToTarget -
            #                        current_distance)
            distance_rate = -current_distance + 4
            angle_reward = (math.pi - abs(AngleToTarget / 180 * math.pi))**2
            reward = (0.1 * distance_rate + 0.9 * angle_reward)
            self.past_distanceToTarget = current_distance
            self.past_AngleToTarget = AngleToTarget
        print('距离:', current_distance, '角度：', AngleToTarget, '奖赏：', reward)
        return np.asarray(state), reward, done

    def reset(self):
        '''
        复位环境
        '''
        # 其实不需要复位gazebo直接改变机器人和目标点的位置即可
        # self.resetGazebo()

        # 传送机器人到一个随机的位置
        agentX, agentY = self.agentController.teleportfixed()

        # 每次启动都删出目标点，重新设置目标点
        # self.goalCont.deleteModel()
        self.goalCont.respawnModel()
        # 重启仿真观测数据
        # self.unpauseGazebo()
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        # self.pauseGazebo()

        # 获得机器人的数据
        state, isCrash = self.calculateState(laserData, odomData)
        # Inform user
        print("New goal position : " + str(self.targetPointX) + " , " +
              str(self.targetPointY))
        self.stateSize = len(state)
        return np.asarray(state)  # Return state
