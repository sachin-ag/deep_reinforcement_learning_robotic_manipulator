from env import ClutteredPushGrasp
from robot import UR5Robotiq85
from rl import DDPG
import pybullet as p
import numpy as np
import sys

MAX_EPISODES = 1000
MAX_EP_STEPS = 100

rl = DDPG(4, 31, [-0.5, 0.5], int((MAX_EPISODES*MAX_EP_STEPS)/10))

def simulate(filename):
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
    
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0, 0), (0, 0, 0)), vis=True)
    s = env.reset()
    rl.restore()
    error_ = 0
    episodes = 0
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
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            r_ += r

        pos = env.robot.get_ee_pos()
        error = -np.sqrt((env.goal[0]-pos[0])**2 + (env.goal[1]-pos[1])**2 + (env.goal[2]-pos[2])**2)
        error_ += error
        print('\nGoal:', env.goal)
        print('Position:', pos)
        print('Reward:', r_)
        print('Error:', error)

    print("\n\nTotal Error:", error_)
    env.close()
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide the path file on command line.")
    else:
        filename = sys.argv[1]
        simulate(filename)