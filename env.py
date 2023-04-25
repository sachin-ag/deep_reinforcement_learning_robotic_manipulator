import numpy as np
import pybullet as p
import pybullet_data

THRESHOLD = 0.1
ON_GOAL_THRESHOLD = 25
STEP_SIMULATIONS = 30

class ClutteredPushGrasp:

    def __init__(self, robot, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        self.camera = camera
        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        self.on_goal = 0
        for joint_id in self.robot.arm_controllable_joints[:-2]:
            p.changeDynamics(self.robot.id, joint_id, jointLowerLimit=-float('inf'), jointUpperLimit=float('inf'))

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

    def step(self, action, control_method='joint'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        action = np.append(action, [0, 0], 0)
        action = action + self.robot.get_joint_angles()
        self.prev_joint_angles = self.robot.get_joint_angles()
        self.robot.move_ee(action, control_method)
        for _ in range(STEP_SIMULATIONS):
            self.step_simulation()
        self.curr_joint_angles = self.robot.get_joint_angles()
        reward, done = self.update_reward_ddpg()
        return self.generate_state(done), reward, done

    def set_goal(self, goal):
        self.goal = goal

    def set_random_goal(self):
        x = np.random.uniform(.25, .75)
        y = np.random.uniform(.25, .75)
        z = np.random.uniform(.25, .75)
        x *= np.random.choice([-1, 1])
        y *= np.random.choice([-1, 1])
        self.goal = [round(x, 2), round(y, 2), round(z, 2)]

    def update_reward_ddpg(self):
        rc = 0
        ra = 0
        rd = 0
        pos = self.robot.get_ee_pos()
        dist = np.sqrt((self.goal[0]-pos[0])**2 +
                     (self.goal[1]-pos[1])**2 + (self.goal[2]-pos[2])**2)
        if dist <= THRESHOLD:
            rd = 100 * (1 - (dist/THRESHOLD))**.5
        else:
            rd = (1 - (dist/THRESHOLD))
        done = False
        if dist <= THRESHOLD:
            # r += 1
            self.on_goal += 1
            self.x += rd
            if self.on_goal >= ON_GOAL_THRESHOLD:
                ra += self.x
                done = True
        else:
            self.on_goal = 0
            self.x = 0
            
        for prev_angle, curr_angle in zip(self.prev_joint_angles[:-2], self.curr_joint_angles[:-2]):
            rc -= abs(curr_angle - prev_angle) / 4
            # if abs(curr_angle - prev_angle) > .785: # pi/4
                # rc -= 1

        r = (ra + rc + rd)
        return r, done

    def generate_state(self, done=False):
        ee_pos = self.robot.get_ee_pos()
        goal = self.goal
        joints_pos = self.robot.get_joint_pos()[:-2]
        s = []
        for joint_pos in joints_pos:
            s.extend([joint_pos[0], joint_pos[1], joint_pos[2], (joint_pos[0] -
                     goal[0]), (joint_pos[1] - goal[1]), (joint_pos[2] - goal[2])])
        s.extend([ee_pos[0], ee_pos[1], ee_pos[2], (ee_pos[0] -
                 goal[0]), (ee_pos[1] - goal[1]), (ee_pos[2] - goal[2])])
        s.append(1) if done else s.append(0)
        return np.array(s)

    def reset(self):
        self.x = 0
        self.on_goal = 0
        self.set_random_goal()
        return self.generate_state()

    def close(self):
        p.disconnect(self.physicsClient)
