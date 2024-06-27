import csv
import gym
from gym import spaces
import pybullet as p
import numpy as np

DEFAULT_PORT = 9543
JOINT_FORCE = 400
TABLE_HEIGHT = 1.0
TABLE_THICKNESS = 0.08
TABLE_LENGTH = 2.4
TABLE_WIDTH = 1.4
DT = 1.0/50
INITIAL_CAMERA_ANGLE = 90.0
INITIAL_CAMERA_DISTANCE = 3.0
NO_DEADLINE = 100.0*365.0*86400.0
JOINTS = 12
TEXT_HEIGHT = 2.5
TEXT_POS = 0.5*TABLE_LENGTH+0.5
STOPPED_SPEED_THRESHOLD = 0.01
STOPPED_TIME_THRESHOLD = 1.0
DIST_THRESHOLD = 3*TABLE_LENGTH
STATE_DIMENSION = 12
BALL_SERVICE_HEIGHT = TABLE_HEIGHT+1.0
FONT_SIZE = 2.0

csvfile = open('data.csv', 'w', newline='')
spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['robot_pos', 'target_pos'])

class RobotTableEnv(gym.Env):
    def __init__(self):
        super(RobotTableEnv, self).__init__()

        # PyBullet setup
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        self.robot = p.loadURDF(
            'robot.urdf', [0, -0.85 - 0.5 * TABLE_LENGTH, 0.8])
        self.num_joints = p.getNumJoints(self.robot)

        # Random position within the pallets range
        self.box_pos = [np.random.uniform(0),
                        np.random.uniform(-0.5-0.5*TABLE_LENGTH), np.random.uniform(2.0, 2.5)]
        self.box = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        self.box_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-0.3, -0.8, -np.inf, -np.pi/2, -np.inf, -3*np.pi /
                         4, -np.inf, -3*np.pi/4, -np.inf, -3*np.pi/4, -np.inf, -np.inf]),
            high=np.array([0.3, 0.8, np.inf, np.pi/2, np.inf, 3*np.pi/4,
                           np.inf, 3*np.pi/4, np.inf, 3*np.pi/4, np.inf, np.inf]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSION,), dtype=np.float32)

        # Constants
        self.dt = 1.0 / 50
        self.max_steps = 1000  # Maximum number of steps per episode
        self.current_step = 0  # Initialize current step counter

        # Camera settings
        self.camera_angle = INITIAL_CAMERA_ANGLE
        self.camera_distance = INITIAL_CAMERA_DISTANCE

        self.joint_force = 400

        # Initialize state
        self.state = np.zeros(STATE_DIMENSION)

    def update_gui(self):
        keys = p.getKeyboardEvents()
        if self.pressed(keys, p.B3G_LEFT_ARROW):
            self.camera_angle -= DT*25.0
        if self.pressed(keys, p.B3G_RIGHT_ARROW):
            self.camera_angle += DT*25.0
        if self.pressed(keys, p.B3G_UP_ARROW):
            self.camera_distance = max(1.5, self.camera_distance
                                       - DT*0.5)
        if self.pressed(keys, p.B3G_DOWN_ARROW):
            self.camera_distance = min(5.0, self.camera_distance
                                       + DT*0.5)
        if self.pressed(keys, ord(' ')):
            self.camera_angle = INITIAL_CAMERA_ANGLE
            self.camera_distance = INITIAL_CAMERA_DISTANCE
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_angle,
            cameraPitch=-30.0,
            cameraTargetPosition=[0.0, 0.0, 1.0])

    def get_paddle_position_and_normal(self):
        rob = self.robot
        pos = [0.0] * 3
        nor = [0.0] * 3
        ls = p.getLinkState(rob, 11)
        pos[0:3] = ls[0][0:3]
        quat = ls[1]
        mat = p.getMatrixFromQuaternion(quat)
        nor[0] = mat[2]
        nor[1] = mat[5]
        nor[2] = mat[8]
        return pos, nor

    def step(self, action):
        # Apply action to robot joints
        self.update_gui()
        self.update_gui()
        self._set_robot_joints(action)

        self.current_step += 1
        done = False
        if self.current_step >= self.max_steps:
            done = True

        # Step simulation
        p.stepSimulation()

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done (if needed)

        return observation, reward, done, {}

    def pressed(self, keys, k):
        return keys.get(k, 0) & p.KEY_IS_DOWN

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def reset(self):
        # Reset simulation
        p.resetSimulation()

        self.current_step = 0
        # Reload robot
        self.robot = p.loadURDF(
            'robot.urdf', [0, -0.85 - 0.5 * TABLE_LENGTH, 0.8])
        self.num_joints = p.getNumJoints(self.robot)

        print(f"Number of joints: {self.num_joints}")
        # Random position within the pallets range
        self.box_pos = [np.random.uniform(1, -2),
                        np.random.uniform(-1.0, -2.0), np.random.uniform(2.0, 2.5)]
        self.box = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        self.box_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        self.box_body = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=self.box, baseVisualShapeIndex=self.box_visual, basePosition=self.box_pos)
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSION,), dtype=np.float32)
        # Return initial observation
        return self._get_observation()

    def _set_robot_joints(self, action):
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot, i, p.POSITION_CONTROL, targetPosition=action[i], force=self.joint_force)

    def _get_observation(self):
        # Fetch robot joint states
        joint_states = [p.getJointState(self.robot, i)[0]
                        for i in range(self.num_joints)]

    # Combine robot joint states with other relevant information to form observation
        observation = np.array(joint_states)

    # Ensure observation array has a consistent shape
        observation = observation.flatten()

        return observation

    def _calculate_reward(self):
        # Calculate reward based on distance between robot and target
        robot_pos, _ = self.get_paddle_position_and_normal()

        # Set the target position here (e.g., center of the table)
        target_pos = self.box_pos
        spamwriter.writerow([robot_pos, target_pos])
        distance_to_target = np.linalg.norm(
            np.array(robot_pos) - np.array(target_pos))

        # Define a reward function that encourages the robot to move closer to the target
        reward = -(distance_to_target * distance_to_target)

        print(f"reward: {reward}")

        return reward
