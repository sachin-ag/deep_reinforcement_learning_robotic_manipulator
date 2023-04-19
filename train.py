from env import ClutteredPushGrasp
from robot import UR5Robotiq85
from rl import DDPG
import pybullet as p
import numpy as np
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_EPISODES = 1000
MAX_EP_STEPS = 100


rl = DDPG(4, 31, [-0.5, 0.5], int((MAX_EPISODES*MAX_EP_STEPS)/20))


def train():
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0, 0), (0, 0, 0)))
    f1 = open("./results/accuracy.txt", 'w')
    f2 = open("./results/rewards.txt", 'w')
    f3 = open("./results/q_values.txt", 'w')
    acc = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        ep_q = 0.
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s_, r, done = env.step(a, 'joint')
            rl.store_transition(s, a, r, s_)
            ep_r += r
            ep_q += rl.get_q_value(s, a)
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
                f3.write("%f\n" % ep_q)
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
    
train()