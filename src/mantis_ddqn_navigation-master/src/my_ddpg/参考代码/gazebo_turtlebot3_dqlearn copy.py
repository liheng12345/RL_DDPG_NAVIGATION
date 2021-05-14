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
import tf
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

    def teleportRandom(self):
        '''
        传送机器人位置
        返回[posX,posY]
        '''

        model_state_msg = ModelState()
        model_state_msg.model_name = self.agent_model_name

        if SELECT_MAP == "maze1":
            # maze 1
            xy_list = [[-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
                       [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
                       [5.5, -1.5], [5.5, -0.5], [5.5, 0.5], [5.5, 1.5]]
        elif SELECT_MAP == "maze2":
            # maze 2
            xy_list = [
                [-1.5, -1.5],
                [-0.5, -1.5],
                [-1.5, -0.5],
                [-0.5, 1.5],
                [1.5, 0.5],
                [2.5, 2.5],
                [2.5, 3.5],
                [1.5, 3.5],
            ]
        else:
            # maze 3
            xy_list = [
                [0.5, 0.5],
                [1.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [-0.5, -0.5],
                [-1.5, -0.5],
                [-1.5, -1.5],
                [0.5, -0.5],
                [0.5, -1.5],
                [1.5, -1.5],
                [-1.5, 0.5],
                [-0.5, 1.5],
                [-1.5, 1.5],
            ]

        # 随机选择机器人的位置
        pose = Pose()
        pose.position.x, pose.position.y = random.choice(xy_list)

        model_state_msg.pose = pose
        model_state_msg.twist = Twist()

        model_state_msg.reference_frame = "world"

        # Start teleporting in Gazebo
        isTeleportSuccess = False
        for i in range(5):
            if not isTeleportSuccess:
                try:
                    rospy.wait_for_service('/gazebo/set_model_state')
                    telep_model_prox = rospy.ServiceProxy(
                        '/gazebo/set_model_state', SetModelState)
                    telep_model_prox(model_state_msg)
                    isTeleportSuccess = True
                    break
                except Exception as e:
                    rospy.logfatal("设置机器人位置成功 " + str(e))
            else:
                #rospy.logwarn("设置机器人位置失败..." + str(i))
                time.sleep(2)

        if not isTeleportSuccess:
            rospy.logfatal("两次设置机器人位置失败")
            return "Err", "Err"

        return pose.position.x, pose.position.y


class GoalController():
    """
    机器人目标控制
    """
    def __init__(self):
        self.model_path = "../models/gazebo/goal_sign/model.sdf"
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
        isSpawnSuccess = False
        for i in range(5):
            if not self.check_model:  # This used to checking before spawn model if there is already a model
                try:
                    rospy.wait_for_service('gazebo/spawn_sdf_model')
                    spawn_model_prox = rospy.ServiceProxy(
                        'gazebo/spawn_sdf_model', SpawnModel)
                    spawn_model_prox(self.model_name, self.model,
                                     'robotos_name_space', self.goal_position,
                                     "world")
                    isSpawnSuccess = True
                    self.check_model = True
                    break
                except Exception as e:
                    rospy.logfatal("Error when spawning the goal sign " +
                                   str(e))
            else:
                rospy.logwarn("Trying to spawn goal sign ..." + str(i))
                time.sleep(2)

        if not isSpawnSuccess:
            rospy.logfatal("Error when spawning the goal sign")

    def deleteModel(self):
        '''
        在gazebo中删除圈
        '''
        while True:
            if self.check_model:
                try:
                    rospy.wait_for_service('gazebo/delete_model')
                    del_model_prox = rospy.ServiceProxy(
                        'gazebo/delete_model', DeleteModel)
                    del_model_prox(self.model_name)
                    self.check_model = False
                    break
                except Exception as e:
                    rospy.logfatal("Error when deleting the goal sign " +
                                   str(e))
            else:
                break

    def calcTargetPoint(self):
        """
        随机返回一个目标点在gazebo中
        """
        self.deleteModel()
        # Wait for deleting
        time.sleep(0.5)

        if SELECT_MAP == "maze1":
            # maze 1
            goal_xy_list = [[-1.5, 0.5], [-1.5, 1.5], [-0.5, 0.5], [-0.5, 1.5],
                            [0.5, -0.5], [0.5, -1.5], [2.5, -0.5], [2.5, 0.5],
                            [5.5, -1.5], [5.5, -0.5], [5.5, 0.5], [5.5, 1.5]]
        elif SELECT_MAP == "maze2":
            # maze 2
            goal_xy_list = [
                [-1.5, -1.5],
                [-0.5, -1.5],
                [-1.5, -0.5],
                [-0.5, 1.5],
                [1.5, 0.5],
                [2.5, 2.5],
                [2.5, 3.5],
                [1.5, 3.5],
            ]
        else:
            # maze 3
            goal_xy_list = [
                [0.5, 0.5],
                [1.5, 0.5],
                [0.5, 1.5],
                [1.5, 1.5],
                [-0.5, -0.5],
                [-1.5, -0.5],
                [-1.5, -1.5],
                [0.5, -0.5],
                [0.5, -1.5],
                [1.5, -1.5],
                [-1.5, 0.5],
                [-0.5, 1.5],
                [-1.5, 1.5],
            ]

        # Check last goal position not same with new goal
        while True:
            self.goal_position.position.x, self.goal_position.position.y = random.choice(
                goal_xy_list)

            if self.last_goal_x != self.goal_position.position.x:
                if self.last_goal_y != self.goal_position.position.y:
                    break

        # Spawn goal model
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        # Inform user
        print("New goal position : " + str(self.goal_position.position.x) +
              " , " + str(self.goal_position.position.y))

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
        self.actionSize = 5  # 机器人动作个数
        self.targetDistance = 0  # 离目标点的距离

        self.targetPointX = 0  # Target Pos X
        self.targetPointY = 0  # Target Pos Y

        # 默认意味机器人到达了目标点，再开始重新复位环境
        self.isTargetReached = True

        # 定义目标点生成器
        self.goalCont = GoalController()
        #定义机器人控制器
        self.agentController = AgentPosController()

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
        try:
            laserData = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            return laserData
        except Exception as e:
            rospy.logfatal("获取雷达数据失败 " + str(e))

    def getOdomData(self):
        '''
        返回机器人的位置和朝向
        '''
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
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(
                quatTuple)
            robotX = odomData.position.x
            robotY = odomData.position.y
            return yaw, robotX, robotY

        except Exception as e:
            rospy.logfatal("获取odom失败 " + str(e))

    def calcHeadingAngle(self, targetPointX, targetPointY, yaw, robotX,
                         robotY):
        '''
        计算机器人和目标点的朝向
        '''
        targetAngle = math.atan2(targetPointY - robotY, targetPointX - robotX)

        heading = targetAngle - yaw
        if heading > math.pi:
            heading -= 2 * math.pi

        elif heading < -math.pi:
            heading += 2 * math.pi

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
        总共是24+4+1
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
            # 雷达数据无穷大的时候强制设置为10
            if np.isinf(laserData[i]):
                laserData[i] = self.laserMaxRange
            if np.isnan(laserData[i]):
                laserData[i] = 0
        # 保留两位小数，四舍五入
        obstacleMinRange = round(min(laserData), 2)
        #找到最小的雷达数据的索引
        obstacleAngle = np.argmin(laserData)

        return laserData + [
            heading, distance, obstacleMinRange, obstacleAngle
        ], isCrash

    def step(self, action):
        '''
        在仿真环境中执行一步，并且暂停仿真，返回机器人状态信息
        '''
        # gazebo不停止
        self.unpauseGazebo()

        # 计算机器人朝向，速度
        maxAngularVel = 1.5
        angVel = ((self.actionSize - 1) / 2 - action) * maxAngularVel / 2
        velCmd = Twist()
        velCmd.linear.x = 0.15
        velCmd.angular.z = angVel

        self.velPub.publish(velCmd)

        # More basic actions
        """
        if action == 0: #快速向左
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 1.6
            self.velPub.publish(velCmd)
        elif action == 1: #向左
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 0.8
            self.velPub.publish(velCmd)
        elif action == 2: #向前
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = 0.0
            self.velPub.publish(velCmd)
        elif action == 3: #向右
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = -0.8
            self.velPub.publish(velCmd)
        elif action == 4: #快速向右
            velCmd = Twist()
            velCmd.linear.x = 0.17
            velCmd.angular.z = -1.6
            self.velPub.publish(velCmd)       
        """

        # 获取观测
        laserData = self.getLaserData()
        odomData = self.getOdomData()

        self.pauseGazebo()

        state, isCrash = self.calculateState(laserData, odomData)

        done = False
        if isCrash:
            done = True

        distanceToTarget = state[-3]

        if distanceToTarget < 0.2:  # Reached to target
            self.isTargetReached = True
        #碰撞以后reward
        if isCrash:
            reward = -150

        elif self.isTargetReached:
            # Reached to target
            rospy.logwarn("到达目标点")
            reward = 200
            # Calc new target point
            self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint(
            )
            self.isTargetReached = False

        else:
            # 没有到达目标点也没有发生碰撞
            yawReward = []
            currentDistance = state[-3]
            heading = state[-4]

            # 计算奖赏
            # 参考 https://emanual.robotis.com/docs/en/platform/turtlebot3/ros2_machine_learning/

            for i in range(self.actionSize):
                angle = -math.pi / 4 + heading + (math.pi / 8 *
                                                  i) + math.pi / 2
                tr = 1 - 4 * math.fabs(0.5 -
                                       math.modf(0.25 + 0.5 * angle %
                                                 (2 * math.pi) / math.pi)[0])
                yawReward.append(tr)

            try:
                distanceRate = 2**(currentDistance / self.targetDistance)
            except Exception:
                print("Overflow err CurrentDistance = ", currentDistance,
                      " TargetDistance = ", self.targetDistance)
                distanceRate = 2**(currentDistance // self.targetDistance)

            reward = ((round(yawReward[action] * 5, 2)) * distanceRate)

        return np.asarray(state), reward, done

    def reset(self):
        '''
        复位环境
        '''
        self.resetGazebo()

        while True:
            # 传送机器人到一个随机的位置
            agentX, agentY = self.agentController.teleportRandom()
            if self.calcDistance(self.targetPointX, self.targetPointY, agentX,
                                 agentY) > self.minCrashRange:
                break
            else:
                rospy.logerr("重新传送机器人")
                time.sleep(2)

        if self.isTargetReached:
            while True:
                self.targetPointX, self.targetPointY = self.goalCont.calcTargetPoint(
                )
                if self.calcDistance(self.targetPointX, self.targetPointY,
                                     agentX, agentY) > self.minCrashRange:
                    self.isTargetReached = False
                    break
                else:
                    rospy.logerr("采用下一个目标点失败，，重新计算下一个目标点")
                    time.sleep(2)

        # 重启仿真观测数据
        self.unpauseGazebo()
        laserData = self.getLaserData()
        odomData = self.getOdomData()
        self.pauseGazebo()
        # 获得机器人的数据
        state, isCrash = self.calculateState(laserData, odomData)
        self.targetDistance = state[-3]
        self.stateSize = len(state)

        return np.asarray(state)  # Return state
