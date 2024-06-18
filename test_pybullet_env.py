import pybullet as p
import pybullet_data as pd
import time
import math

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
JOINTS = 11
TEXT_HEIGHT = 2.5
TEXT_POS = 0.5*TABLE_LENGTH+0.5
STOPPED_SPEED_THRESHOLD = 0.01
STOPPED_TIME_THRESHOLD = 1.0
DIST_THRESHOLD = 3*TABLE_LENGTH
STATE_DIMENSION = 37
BALL_SERVICE_HEIGHT = TABLE_HEIGHT+1.0
FONT_SIZE = 2.0


class Playfield:
    def __init__(self, gui=True):
        self.has_gui = gui
        self.init_pybullet()
        self.load_objects()
        self.box = None
        self.cpoints = []
        self.camera_angle = INITIAL_CAMERA_ANGLE
        self.camera_distance = INITIAL_CAMERA_DISTANCE
        self.finished = False
        self.sim_time = 0.0
        self.update_state_cb = None
        self.next_state_cb = None
        self.update_state_deadline = 0.0
        self.ball_held_position = None
        self.stopped_time = 0.0
        self.hold_ball([0, 0, 2.0])
        self.schedule_start_positions()
        self.set_central_text('waiting for players')

    def load_objects(self):
        self.floor = p.loadURDF('plane.urdf')
        self.table = p.loadURDF('table.urdf', [0, 0,
                                               TABLE_HEIGHT-TABLE_THICKNESS])
        self.ball = p.loadURDF('ball.urdf', [0, 0.1, 2],
                               flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.robot = [None, None]
        self.robot[0] = p.loadURDF('robot.urdf',
                                   [0, -0.85-0.5*TABLE_LENGTH, 0.8])
        rot = p.getQuaternionFromEuler([0, 0, math.pi])
        self.robot[1] = p.loadURDF('robot2.urdf',

                                   [0, +0.85+0.5*TABLE_LENGTH, 0.8], rot)
        self.objects = [self.floor, self.table, self.ball] + self.robot
        for obj in self.objects:
            for j in range(-1, p.getNumJoints(obj)):
                r = 0.95
                if obj == self.table and j == 0:
                    r = 0.1
                elif (obj == self.robot[0] or obj == self.robot[1]):
                    if j <= 1:
                        r = 0.7
                    elif j >= JOINTS-1:
                        r = 1.1
                elif obj == self.floor:
                    r = 0.7
                p.changeDynamics(obj, j, restitution=r, lateralFriction=3.0)
                if j >= 0:
                    p.setJointMotorControl2(obj, j, p.POSITION_CONTROL,
                                            -0.2, force=JOINT_FORCE)
        self.name = ["Player 1", "Player 2"]
        self.text = [None, None, None]
        if self.has_gui:
            self.text[0] = p.addUserDebugText(self.name[0],
                                              [0, -TEXT_POS, TEXT_HEIGHT], textSize=FONT_SIZE)
            self.text[1] = p.addUserDebugText(self.name[1],
                                              [0, +TEXT_POS, TEXT_HEIGHT], textSize=FONT_SIZE)
            self.text[2] = p.addUserDebugText("",
                                              [0, 0.0, TEXT_HEIGHT], textSize=FONT_SIZE)

    def add_box(self, position, half_extents=[0.1, 0.1, 0.1], color=[1, 0, 0, 1]):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

    def update(self):
        self.update_gui()
        self.update_ball()
        if self.update_state_cb:
            if self.sim_time >= self.update_state_deadline:
                self.update_state_cb = None
                if self.next_state_cb:
                    next_cb = self.next_state_cb
                    self.next_state_cb = None
                    next_cb()
            else:
                self.update_state_cb()

        pf = self
        index = 0
        y_dir = 2*index-1
        state = [0.0]*STATE_DIMENSION
        state[0:10] = self.get_robot_joints(index)
        pad = self.get_paddle_position_and_normal(index)
        state[11:14] = self.convert_coordinates(index, pad[0])
        state[14:17] = self.convert_vector(index, pad[1], +1)
        state[17:20] = self.convert_coordinates(index, pf.ball_position)
        state[20:23] = self.convert_vector(index, pf.ball_velocity)

    def convert_coordinates(self, index, vec):
        orig = self.get_player_origin(index)
        if index == 0:
            return vec[0]-orig[0], vec[1]-orig[1], vec[2]-orig[2]
        else:
            return orig[0]-vec[0], orig[1]-vec[1], vec[2]-orig[2]

    def convert_vector(self, index, vec, desired_y_sign=None):
        x, y, z = vec
        if index == 1:
            x = -x
            y = -y
        if desired_y_sign:
            if y*desired_y_sign < 0.0:
                x = -x
                y = -y
                z = -z
        return x, y, z

    def step_simulation(self):
        p.stepSimulation()

    def run(self):
        ref_time = time.time()
        try:
            while not self.finished:
                p.stepSimulation()
                self.cpoints += p.getContactPoints(self.ball)
                p.stepSimulation()
                self.cpoints += p.getContactPoints(self.ball)
                self.update()
                self.sim_time += DT
                now = time.time()
                dt = ref_time+DT-now
                if dt <= 0.0:
                    ref_time = now
                else:
                    ref_time += DT
                    time.sleep(dt)
        finally:
            p.disconnect()

    def update_gui(self):
        if not self.has_gui:
            return
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

    def pressed(self, keys, k):
        return keys.get(k, 0) & p.KEY_IS_DOWN

    def update_ball(self):
        if self.ball_held_position:
            p.resetBasePositionAndOrientation(self.ball,
                                              self.ball_held_position, [1.0, 0.0, 0.0, 0.0])
        po = p.getBasePositionAndOrientation(self.ball)
        pos = po[0]
        self.ball_position = pos
        dist = math.hypot(pos[0], pos[1])
        self.ball_away = (dist > DIST_THRESHOLD)
        bv = p.getBaseVelocity(self.ball)
        v = bv[0]
        self.ball_velocity = v
        self.ball_speed = math.hypot(v[0], v[1], v[2])
        if self.ball_speed > STOPPED_SPEED_THRESHOLD:
            self.stopped_time = 0.0
        else:
            self.stopped_time += DT
        self.ball_stopped = (self.stopped_time > STOPPED_TIME_THRESHOLD)
        cpoints = self.cpoints
        self.cpoints = []
        self.contact_floor = False
        self.contact_table = False
        self.contact_robot = [False, False]
        for cp in cpoints:
            if cp[1] != self.ball:
                continue
            elif cp[2] == self.table and cp[4] == -1:
                self.contact_table = True
            elif cp[2] == self.floor:
                self.contact_floor = True
            elif cp[2] == self.robot[0] and cp[4] > 1:
                self.contact_robot[0] = True
            elif cp[2] == self.robot[1] and cp[4] > 1:
                self.contact_robot[1] = True
            elif cp[2] in self.robot:
                # cp[4]<=1
                self.contact_floor = True

    def set_update_state_callback(self, update_cb, next_cb=None,
                                  duration=NO_DEADLINE):
        self.update_state_cb = update_cb
        self.next_state_cb = next_cb
        self.update_state_deadline = self.sim_time+duration

    def init_pybullet(self):
        if not self.has_gui:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.5*DT)
        p.setAdditionalSearchPath(pd.getDataPath())

    def schedule_start_positions(self, after_cb=None):
        def next_cb1():
            self.set_update_state_callback(self.start_pos2,
                                           after_cb, 0.9)
        self.set_update_state_callback(self.start_pos1,
                                       next_cb1, 0.7)

    def start_pos1(self):
        jp = [0.0]*JOINTS
        jp[0] = -0.3
        jp[2] = math.pi
        self.set_robot_joints(0, jp)
        self.set_robot_joints(1, jp)

    def start_pos2(self):
        jp = get_neutral_joint_position()
        self.set_robot_joints(0, jp)
        self.set_robot_joints(1, jp)

    def quit(self):
        self.finished = True

    def get_robot_joints(self, index):
        jp = [0.0]*JOINTS
        rob = self.robot[index]
        for j in range(JOINTS):
            jp[j] = p.getJointState(rob, j)[0]
        return jp

    def set_robot_joints(self, index, values):
        rob = self.robot[index]
        for j, val in enumerate(values):
            p.setJointMotorControl2(rob, j,
                                    p.POSITION_CONTROL,
                                    val,
                                    force=JOINT_FORCE)

    def get_paddle_position_and_normal(self, index):
        rob = self.robot[index]
        pos = [0.0]*3
        nor = [0.0]*3
        ls = p.getLinkState(rob, JOINTS)
        pos[0:3] = ls[0][0:3]
        quat = ls[1]
        mat = p.getMatrixFromQuaternion(quat)
        nor[0] = mat[2]
        nor[1] = mat[5]
        nor[2] = mat[8]
        return pos, nor

    def set_text(self, index, added=None):
        if not self.has_gui:
            return
        elif index == 2:
            text = added if added else ""
        elif added is not None:
            text = self.name[index]+": "+str(added)
        else:
            text = self.name[index]
        pos = 0.0 if index == 2 else TEXT_POS*(2*index-1)
        z = TEXT_HEIGHT
        if index == 2:
            z += 0.5
        p.addUserDebugText(text, [0.0, pos, z],
                           textSize=FONT_SIZE,
                           replaceItemUniqueId=self.text[index])

    def set_central_text(self, text):
        self.set_text(2, text)

    def set_name(self, index, name):
        self.name[index] = name
        self.set_text(index)

    def hold_ball(self, pos):
        self.ball_held_position = pos

    def throw_ball(self, velocity):
        self.ball_held_position = None
        self.ball_stopped = False
        self.stopped_time = 0.0
        p.resetBaseVelocity(self.ball, velocity)

    def get_player_origin(self, index):
        y = index*2.0-1
        return [0.0, y, TABLE_HEIGHT]

    def reset(self):
        self.hold_ball([0, 0, 2.0])
        self.schedule_start_positions()
        self.box = None

    def get_player_direction_y(self, index):
        return 1-index*2


def get_neutral_joint_position():
    jp = [0.0]*JOINTS
    jp[0] = -0.3
    jp[2] = math.pi
    a = math.pi/3.8
    jp[5] = a
    jp[7] = a
    jp[9] = math.pi/3.5
    jp[10] = math.pi/2
    return jp


if __name__ == '__main__':
    pf = Playfield()

    # Define a callback to change joint positions for both robots
    def change_robot_joints():
        # Get current joint positions
        current_joints_robot_0 = pf.get_robot_joints(0)
        current_joints_robot_1 = pf.get_robot_joints(1)

        # Modify the joint positions as needed
        current_joints_robot_0[0] = 0.5  # Example change for robot 0
        current_joints_robot_1[0] = -0.5  # Example change for robot 1

        # Set the new joint positions
        pf.set_robot_joints(0, current_joints_robot_0)
        pf.set_robot_joints(1, current_joints_robot_1)

    # Schedule the callback to change the joint positions after a certain time
    # Change joints after 2 seconds
    pf.schedule_start_positions()
    p.stepSimulation()
    pf.set_update_state_callback(change_robot_joints, duration=2.0)
    pf.update()
    p.stepSimulation()

    time.sleep(5)

    # Run the simulation
