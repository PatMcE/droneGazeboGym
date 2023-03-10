'''Give credit to the construct and droneAirsim.py'''

import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import math
import time
import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from gym.envs.registration import register
timestep_limit_per_episode = 500#200

from std_srvs.srv import Empty
import cv2 as cv
import sys

from mavros_px4_vehicle.px4_modes import PX4_MODE_OFFBOARD
from mavros_px4_vehicle.px4_offboard_modes import SetPositionWithYawCmdBuilder
from mavros_px4_vehicle.px4_vehicle import PX4Vehicle

register(
		id='DroneGymGazeboEnv-v0',
		entry_point='drone_gym_gazebo_env:DroneGymGazeboEnv',
		max_episode_steps=timestep_limit_per_episode,
	)

class DroneGymGazeboEnv(gym.Env):
	def __init__(self):
		self.drone = PX4Vehicle(auto_connect = True) #drone treated as an object thanks to 'mavros_px4_vehicle'
		self.z = 0.7 #height the drone flys at
		
		self.action_duration = 1 #length of time of moving forward/left/right in a single step (in seconds)
		self.action_space = spaces.Discrete(3) #3 actions = forward/left/right
		self.end_episode_reward = 100 #Value added/subtracted to/from cumulative reward depending if reached goal or not
		self.height = 36 #height want image to be after processing (original image = 480)
		self.width = 48 #width want image to be after processing (original image = 640)
		self.one_image_shape = (1,self.height,self.width) #(channels, height, width)
		self.two_image_shape = (2,self.height,self.width) #using two images (and relative position coordinates) in state space
		self.image_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.two_image_shape, dtype=np.float32)

        #Set starting and goal/desired point:
		self.start_point = Point()
		self.start_point.x = 0.0
		self.start_point.y = 0.0
		self.start_point.z = self.z
		self.desired_point = Point()
		self.desired_point.x = 10.0#10.0
		self.desired_point.y = 0.0
		self.desired_point.z = self.z

        #Set workspace limits:
		self.work_space_x_max = 11.5
		self.work_space_x_min = -1.0
		self.work_space_y_max = 7.0#3.1
		self.work_space_y_min = -7.0#-3.1
		self.work_space_z_max = 3.0
		self.work_space_z_min = -0.1

		#Additional setup:
		self.seed()
		self.episode_num = 0
		self.cumulated_episode_reward = 0
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.unpause_sim()
		rospy.Subscriber("/camera/depth/image_raw", Image, self._front_camera_depth_image_raw_callback)                
		rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._gt_pose_callback)
		self._check_front_camera_depth_image_raw_ready()
		self._check_gt_pose_ready()
		self.pause_sim()
		self._init_env_variables()
		#self.rate = rospy.Rate(30)#10#100

	def unpause_sim(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except rospy.ServiceException as e:
			print ("/gazebo/unpause_physics service call failed")

	def pause_sim(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except rospy.ServiceException as e:
			print ("/gazebo/pause_physics service call failed")

	def resetWorld(self):
		rospy.wait_for_service('/gazebo/reset_world')
		try:
			self.reset_proxy()
		except rospy.ServiceException as e:
			print ("/gazebo/reset_world service call failed")

	def _init_env_variables(self):
		self.cumulated_steps = 0.0
		self.cumulated_reward = 0.0

		gt_pose = self.get_gt_pose()
		self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.pose.position)

		self.prev_obs = np.zeros(self.one_image_shape, dtype=np.uint16)
		self.prev_prev_obs = np.zeros(self.one_image_shape, dtype=np.uint16)

	def _check_front_camera_depth_image_raw_ready(self):
		self.front_camera_depth_image_raw = None
		rospy.logdebug("Waiting for /camera/depth/image_raw to be READY...")
		while self.front_camera_depth_image_raw is None and not rospy.is_shutdown():
			try:
				self.front_camera_depth_image_raw = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=5.0)
				rospy.logdebug("Current /camera/depth/image_raw READY=>")
			except:
				rospy.logerr("Current /camera/depth/image_raw not ready yet, retrying for getting front_camera_depth_image_raw")
		return self.front_camera_depth_image_raw

	def _check_gt_pose_ready(self):
		self.gt_pose = None
		rospy.logdebug("Waiting for /mavros/local_position/pose to be READY...")
		while self.gt_pose is None and not rospy.is_shutdown():
			try:
				self.gt_pose = rospy.wait_for_message("/mavros/local_position/pose", PoseStamped, timeout=5.0)
				rospy.logdebug("Current /mavros/local_position/pose READY=>")
			except:
				rospy.logerr("Current /mavros/local_position/pose not ready yet, retrying for getting gt_pose")
		return self.gt_pose

	def _front_camera_depth_image_raw_callback(self, image):
		# ROS Callback function for the /camera/depth/image_raw topic
		self.front_camera_depth_image_raw = image

	def _gt_pose_callback(self, data):
		self.gt_pose = data

	def get_front_camera_depth_image_raw(self):
		return self.front_camera_depth_image_raw

	def get_gt_pose(self):
		return self.gt_pose

	def forward(self, speed):
		gt_pose = self.get_gt_pose()        
		x = round(gt_pose.pose.position.x, 2)
		y = round(gt_pose.pose.position.y, 2)

		x_cmd = SetPositionWithYawCmdBuilder.build(x = x + round((speed),2), y = y, z = self.z)
		self.drone.set_pose2d(x_cmd)

		init_time = time.time()
		return init_time

	def left_or_right(self, speed, direction):#dir = 1 for left
		gt_pose = self.get_gt_pose()        
		x = round(gt_pose.pose.position.x, 2)
		y = round(gt_pose.pose.position.y, 2)

		y_cmd = SetPositionWithYawCmdBuilder.build(x = x, y = y + round((direction*speed),2), z = self.z)
		self.drone.set_pose2d(y_cmd)

		init_time = time.time()
		return init_time

	def _set_action(self, action):
		prev_pose = self.get_gt_pose()

		if action == 0:
			rospy.loginfo("> FORWARDS")
			# Move forward at 1m/s for self.action_duration seconds
			start = self.forward(1)
			while self.action_duration > time.time() - start:
				pass

		if action == 1:
			rospy.loginfo("> LEFT")
            # Move left at 1m/s for self.action_duration seconds
			start = self.left_or_right(1, 1)#second 1 indicates move left (not right)
			while self.action_duration > time.time() - start:
				pass

		if action == 2:
			rospy.loginfo("> RIGHT")
            # Move right at 1m/s for self.action_duration seconds
			start = self.left_or_right(1, -1)#-1 indicates move right (not left)
			while self.action_duration > time.time() - start:
				pass

		return prev_pose

	#From imgmsg_to_cv2() at https://github.com/ros-perception/vision_opencv/blob/rolling/cv_bridge/python/cv_bridge/core.py:
	def depth_imgmsg_to_cv2(self, img_msg):#, desired_encoding='passthrough'):
		'''im_msg.encoding of depth image = "32FC1" (single channel float 32 image)'''
		#dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
		#dtype = np.dtype(dtype)
		dtype = np.dtype('uint16')#('f8')
		dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

		#Assuming n_channels = 1:
		im = np.ndarray(shape=(img_msg.height, int(img_msg.step/dtype.itemsize)), dtype=dtype, buffer=img_msg.data)
		im = np.ascontiguousarray(im[:img_msg.height, :img_msg.width])

		# If the byte order is different between the message and the system.
		if img_msg.is_bigendian == (sys.byteorder == 'little'):
			im = im.byteswap().newbyteorder()

		return im

	def preprocess_image(self, image):
		'''help from https://stackoverflow.com/questions/47751323/get-depth-image-in-grayscale-in-ros-with-imgmsg-to-cv2-python'''

		cv_image = self.depth_imgmsg_to_cv2(image)#, desired_encoding="passthrough")
		#rospy.logwarn("cv_image: shape = " + str(cv_image.shape) + " and type = " + str(type(cv_image)))
				
		# Normalize the depth image to fall between 0 (black) and 1 (white)
		#cv_image_norm = cv_image / 255.0#cv.normalize(cv_image, cv_image, 0, 1, cv.NORM_MINMAX)
		#rospy.logwarn("cv_image_norm shape = " + str(cv_image_norm.shape) + " and type = " + str(type(cv_image_norm)))
		
		# Resize to the desired size
		cv_image_resized = cv.resize(cv_image, (self.one_image_shape[2], self.one_image_shape[1]), interpolation = cv.INTER_CUBIC)
		#rospy.logwarn("cv_image_resized shape = " + str(cv_image_resized.shape) + " and type = " + str(type(cv_image_resized)))
		
		cv_image_reshaped = cv_image_resized.reshape((self.one_image_shape[1], self.one_image_shape[2], self.one_image_shape[0]))
		#rospy.logwarn("cv_image_reshaped shape = " + str(cv_image_reshaped.shape) + " and type = " + str(type(cv_image_reshaped)))

		cv_image_transposed = cv_image_reshaped.transpose(2, 0, 1)

		return cv_image_transposed

	def _get_obs(self):
		image = self.get_front_camera_depth_image_raw()
		obs = self.preprocess_image(image)
		gt_pose = self.get_gt_pose()

		combined_obs = np.concatenate((obs, self.prev_prev_obs), axis=0)

		obs_dict = {
			'image': combined_obs,
			'xyz': [round(self.desired_point.x - gt_pose.pose.position.x, 2),
			round(self.desired_point.y - gt_pose.pose.position.y, 2),
			round(self.desired_point.z - gt_pose.pose.position.z, 2)]
		}

		self.prev_prev_obs = self.prev_obs
		self.prev_obs = obs

		return obs_dict

	def is_in_desired_position(self, current_position, epsilon=0.5):
		"""
		It return True if the current position is similar to the desired poistion
		"""

		is_in_desired_pos = False


		x_pos_plus = self.desired_point.x + epsilon
		x_pos_minus = self.desired_point.x - epsilon
		y_pos_plus = self.desired_point.y + epsilon
		y_pos_minus = self.desired_point.y - epsilon

		x_current = current_position.x
		y_current = current_position.y

		x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
		y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

		is_in_desired_pos = x_pos_are_close and y_pos_are_close

		return is_in_desired_pos

	def get_distance_from_desired_point(self, current_position):
		a = np.array((current_position.x, current_position.y, current_position.z))
		b = np.array((self.desired_point.x, self.desired_point.y, self.desired_point.z))

		distance = np.linalg.norm(a - b)

		return distance

	def is_inside_workspace(self, current_position):
		is_inside = False

		if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
			if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
				if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
					is_inside = True

		return is_inside

	def drone_has_collided(self, roll, pitch, yaw):
		"""
		When the drone moves left, right or forward its roll, pitch and yaw remain at approximatley 0.
		If the roll/pitch/yaw deviates much from 0 we know it has hit an object
		"""
		has_collided = True

		self.max_roll = 0.25
		self.max_pitch = 0.25
		self.max_yaw = 0.1

		if roll > -1*self.max_roll and roll <= self.max_roll:
			if pitch > -1*self.max_pitch and pitch <= self.max_pitch:
				if yaw > -1*self.max_yaw and yaw <= self.max_yaw:
					has_collided = False

		return has_collided

	def _is_done(self, observations):
		current_position = Point()
		current_position.x = round(self.desired_point.x - observations['xyz'][0], 2)
		current_position.y = round(self.desired_point.y - observations['xyz'][1], 2)
		current_position.z = round(self.desired_point.z - observations['xyz'][2], 2)
		gt_pose = self.get_gt_pose()
		roll, pitch, yaw = self.euler_from_quaternion(gt_pose.pose.orientation.x, gt_pose.pose.orientation.y, gt_pose.pose.orientation.z, gt_pose.pose.orientation.w)
		rospy.loginfo("pose = " + str(current_position.x) + ", " + str(current_position.y) + ", " + str(current_position.z))

		is_inside_workspace_now = self.is_inside_workspace(current_position)
		has_drone_collided = self.drone_has_collided(roll, pitch, yaw)
		has_reached_des_point = self.is_in_desired_position(current_position, 0.5)

		if (has_drone_collided):
			rospy.logwarn("drone has collided")
			gt_pose = self.get_gt_pose()        
			x = round(gt_pose.pose.position.x, 2)
			y = round(gt_pose.pose.position.y, 2)
			z = round(gt_pose.pose.position.z, 2)
			rospy.logwarn("roll, pitch, yaw = " + str(roll) + str(pitch) + str(yaw))

		episode_done = not(is_inside_workspace_now) or has_drone_collided or has_reached_des_point

		return episode_done	

	def _compute_reward(self, observations, episode_done):
		current_position = Point()
		current_position.x = self.desired_point.x - observations['xyz'][0]
		current_position.y = self.desired_point.y - observations['xyz'][1]
		current_position.z = self.desired_point.z - observations['xyz'][2]

		distance_from_des_point = self.get_distance_from_desired_point(current_position)

		if not episode_done:
			reward = -1 + (self.previous_distance_from_des_point - distance_from_des_point)

		else:
			if self.is_in_desired_position(current_position):
				reward = self.end_episode_reward
				rospy.logwarn("##############")
				rospy.logwarn("in desired pos")
				rospy.logwarn("##############")
			else:
				reward = -1*self.end_episode_reward

		self.previous_distance_from_des_point = distance_from_des_point
		self.cumulated_reward += reward
		self.cumulated_steps += 1

		return reward

	### Env methods start ###
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		rospy.logwarn("seed = " + str(seed))
		return [seed]

	def step(self, action):
		#rospy.logwarn("##### EPISODE NUMBER = " + str(self.episode_num) + " #####")
		self.unpause_sim()
		self._set_action(action)
		self.pause_sim()

		obs = self._get_obs()
		#rospy.loginfo("obs['xyz'] = " + str(obs['xyz']))
		done = self._is_done(obs)
		info = {}
		reward = self._compute_reward(obs, done)
		self.cumulated_episode_reward += reward
		#self.rate.sleep()

		return obs, reward, done, info

	def reset(self):
		rospy.logwarn('start reset')
		#self.rate.sleep()

		#First call of reset, arm drone (no need to move to start point as already there):
		if (self.drone.is_armed() == False):
			rospy.loginfo("Arm drone")
			self.unpause_sim()
			self.drone.arm()
			self.drone.wait_for_status(self.drone.is_armed, True, 2)
			self.pause_sim()
			rospy.loginfo("End arm drone")
		else:#all other calls move to start point (no need to arm drone as already armed)
			gt_pose = self.get_gt_pose()        
			x = round(gt_pose.pose.position.x, 2)
			y = round(gt_pose.pose.position.y, 2)
			z = round(gt_pose.pose.position.z, 2)

			self.unpause_sim()
			rospy.loginfo("send to x-1.0, y, z")
			cmd = SetPositionWithYawCmdBuilder.build(x=x-1.0, y=y, z=z)
			self.drone.set_pose2d(cmd)
			rospy.sleep(10)

			rospy.logwarn("getting x/y before")
			gt_pose = self.get_gt_pose()        
			x = round(gt_pose.pose.position.x, 2)
			y = round(gt_pose.pose.position.y, 2)
			rospy.logwarn("getting x/y after")

			rospy.loginfo("send to x, y, " + str(self.work_space_z_max + 2.0))
			cmd = SetPositionWithYawCmdBuilder.build(x=x, y=y, z=self.work_space_z_max + 2.0)
			self.drone.set_pose2d(cmd)
			rospy.sleep(4)

			rospy.loginfo("send to " + str(self.start_point.x) + ", " + str(self.start_point.y) + ", " + str(self.work_space_z_max + 2.0))
			cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y=self.start_point.y, z=self.work_space_z_max + 2.0)
			self.drone.set_pose2d(cmd)
			rospy.sleep(4)
			
			#rospy.loginfo("reset")
			#self.resetWorld()
			#rospy.sleep(5)

			#rospy.loginfo("send to " + str(self.start_point.x) + ", " + str(self.start_point.y) + ", 0.5")
			#cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y=self.start_point.y, z=0.5)
			#self.drone.set_pose2d(cmd)
			#rospy.sleep(5)

			self.pause_sim()

		self.takeoff_drone()

		self._init_env_variables()
		self._update_episode()
		obs = self._get_obs()

		return obs

	def render(self, mode="human"):
		image = self._get_obs()['image']
		cv.imshow("Image window 1", image[0,:,:])#_resize)#image.astype(np.uint8))
		cv.imshow("Image window 2", image[1,:,:])
		cv.waitKey(1)

	def close(self):
		rospy.logdebug("Closing Environment from drone_gazebo.py")
		rospy.signal_shutdown("Closing Environment from drone_gazebo.py")
    ### ENV methods end ###

	def takeoff_drone(self):
		rospy.loginfo("Sending the set position commands.")
		self.unpause_sim()
		cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y = self.start_point.y, z = self.z)
		self.drone.set_pose2d(cmd)
		rospy.sleep(7)

		rospy.loginfo("Changing to offboard mode.")
		self.drone.set_mode(PX4_MODE_OFFBOARD)
		rospy.sleep(7)
		self.pause_sim()

	def land_disconnect_drone(self):
		rospy.loginfo("Land drone")
		self.unpause_sim()
		self.drone.land(block=True)
		#if self.drone.is_armed():
		#	self.drone.disarm()
		#self.drone.disconnect()
		#self.pause_sim()
		#self.close()

	def _update_episode(self):
		self.episode_num += 1
		self.cumulated_episode_reward = 0

	def euler_from_quaternion(self, x, y, z, w):
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(t3, t4)

		return roll, pitch, yaw #radians