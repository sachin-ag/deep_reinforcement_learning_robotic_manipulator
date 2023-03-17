import numpy as np
import pybullet as p
import pybullet_data


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
        i=0
        for joint_id in self.robot.arm_controllable_joints:
            if i==4:
                break
            p.changeDynamics(self.robot.id, joint_id, jointLowerLimit=-float('inf'), jointUpperLimit=float('inf'))
            i+=1

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
        self.robot.move_ee(action, control_method)
        for _ in range(30):
            self.step_simulation()
        reward, done = self.update_reward_ddpg()
        return self.generate_state(done), reward, done

    def set_goal(self, goal):
        self.goal = goal

    def set_random_goal(self):
        x = np.random.uniform(.25, .75)
        y = np.random.uniform(.25, .75)
        x *= np.random.choice([-1, 1])
        y *= np.random.choice([-1, 1])
        z = np.random.uniform(0.25, 0.75)
        self.goal = [round(x, 2), round(y, 2), round(z, 2)]

    def update_reward_ddpg(self):
        pos = self.robot.get_ee_pos()
        r = -np.sqrt((self.goal[0]-pos[0])**2 +
                     (self.goal[1]-pos[1])**2 + (self.goal[2]-pos[2])**2)
        done = False
        if (self.goal[0] - 0.05 < pos[0] < self.goal[0] + 0.05) and (
                self.goal[1] - 0.05 < pos[1] < self.goal[1] + 0.05) and (
                self.goal[2] - 0.1 < pos[2] < self.goal[2] + 0.1):
            r += 1
            self.on_goal += 1
            if self.on_goal >= 20:
                done = True
        else:
            self.on_goal = 0
        return r, done

    def generate_state(self, done=False):
        ee_pos = self.robot.get_ee_pos()
        goal = self.goal
        joints_pos = self.robot.get_joint_pos()
        s = []
        i = 0
        for joint_pos in joints_pos:
            i += 1
            if i > 4:
                break
            s.extend([joint_pos[0], joint_pos[1], joint_pos[2], (joint_pos[0] -
                     goal[0]), (joint_pos[1] - goal[1]), (joint_pos[2] - goal[2])])
        s.extend([ee_pos[0], ee_pos[1], ee_pos[2]/1, (ee_pos[0] -
                 goal[0]), (ee_pos[1] - goal[1]), (ee_pos[2] - goal[2])])
        if done:
            s.append(1)
        else:
            s.append(0)
        # print(s)
        return np.array(s)

    # def reset_box(self):
    #     p.setJointMotorControl2(self.boxID, 0, p.POSITION_CONTROL, force=1)
    #     p.setJointMotorControl2(self.boxID, 1, p.VELOCITY_CONTROL, force=0)

    def reset(self):
        self.on_goal = 0
        self.set_random_goal()
        self.robot.reset()
        # self.reset_box()
        return self.generate_state()

    def close(self):
        p.disconnect(self.physicsClient)
