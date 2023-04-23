from env import ClutteredPushGrasp, THRESHOLD
from robot import UR5Robotiq85
from td3 import TD3
import pybullet as p
import numpy as np
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MAX_EPISODES = 1000
MAX_EP_STEPS = 100

rl = TD3(31, 4, .5)

def simulate(filename, vis=True):
    points = []
    colors = []

    with open(filename) as f:
        point = f.readline()
        while point != '':
            x, y, z = point.split()
            points.append([float(x), float(y), float(z)])
            colors.append([240./255, 1./255, 1./255])
            point = f.readline()
        f.close()
    
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0, 0), (0, 0, 0)), vis)
    s = env.reset()
    rl.load("checkpoint/td3")
    error_ = 0
    episodes = 0
    completed = 0.
    p.addUserDebugPoints(points, colors, 5)

    for point in points:
        steps = 0
        done = False
        episodes += 1
        r_ = 0
        env.set_goal(point)

        while not done and steps < 100:
            steps += 1
            env.step_simulation()
            a = rl.select_action(s)
            s, r, done = env.step(a)
            r_ += r

        pos = env.robot.get_ee_pos()
        dist = np.sqrt((env.goal[0]-pos[0])**2 + (env.goal[1]-pos[1])**2 + (env.goal[2]-pos[2])**2)
        if dist <= THRESHOLD:
            completed += 1.
        error_ -= dist
        print('\nGoal:', env.goal)
        print('Position:', pos)
        print('Reward:', r_)
        print('Error:', -dist)

    print("\n\nTotal Error:", error_)
    print("Tracking accuracy:", (completed/episodes))
    env.close()
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide the path file on command line.")
    else:
        vis = True
        filename = sys.argv[1]
        if len(sys.argv) == 3:
            vis = False
        simulate(filename, vis)