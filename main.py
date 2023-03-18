from env import ClutteredPushGrasp
from robot import UR5Robotiq85
from rl import DDPG
import pybullet as p
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_EPISODES = 1000
MAX_EP_STEPS = 100


rl = DDPG(4, 31, [-0.5, 0.5], int((MAX_EPISODES*MAX_EP_STEPS)/10))


def train():
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0, 0), (0, 0, 0)))
    f1 = open("./accuracy.txt", 'w')
    f2 = open("./rewards.txt", 'w')
    acc = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a, 'joint')
            rl.store_transition(s, a, r, s_)
            ep_r += r
            if rl.memory_full:
                rl.learn()
            s = s_
            if done or j == MAX_EP_STEPS-1:
                if done:
                    acc = 0.01 + 0.99*acc
                else:
                    acc = 0.99*acc
                f1.write("%f\n" % acc)
                f2.write("%f\n" % ep_r)
                print('\nEp: %i | %s | r: %.1f | acc:' %
                      (i, '----' if not done else 'done', ep_r), round(acc, 2))
                break
        print('Goal:', env.goal, '\nFinal_pos:', env.robot.get_ee_pos())
        if i % 5000 == 0:
            rl.save()
    f1.close()
    f2.close()
    rl.save()
    env.close()


def simulate(filename):
    rl.restore()
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0, 0), (0, 0, 0)), vis=True)
    s = env.reset()
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
    episodes = 0
    p.addUserDebugPoints(points, colors, 5)

    for point in points:
        steps = 0
        done = False
        episodes += 1
        r = 0
        env.set_goal(point)

        while not done and steps < 100:
            steps += 1
            env.step_simulation()
            a = rl.choose_action(s)
            s, r, done = env.step(a)

        print('Goal:', env.goal, '\nFinal_pos:', env.robot.get_ee_pos())
    env.close()


if __name__ == "__main__":
    print("Options:\n0 -> Train\n1 -> Simulate using a file with coordinates")
    inp = input("enter 0 or 1: ")
    if inp == "0":
        train()
    elif inp == "1":
        filename = input("Enter filename: ")
        simulate(filename)
    else:
        print("Invalid option.\n")
