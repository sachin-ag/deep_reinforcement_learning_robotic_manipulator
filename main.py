"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG

MAX_EPISODES = 1000
MAX_EP_STEPS = 200

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    f = open("./accuracy.txt", 'w')
    acc = 0
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_

            if done or j == MAX_EP_STEPS-1:
                if done:
                    acc = 0.01 + 0.99*acc
                else:
                    acc = 0.99*acc
                f.write("%f\n" % acc)
                print('Ep: %i | %s | ep_r: %.1f | step: %i | acc:' % (i, '---' if not done else 'done', ep_r, j), acc)
			    # print(a)
                break
    f.close()
    rl.save()


def eval1():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)

def eval2():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    print("Note: 0<x<600 & 0<y<300.")
    while True:
        inp = input("input x and y coordinates separated by space: ")
        


print("Options:\n1. Train\n2. Simulate using cursor\n3. Simulate using given points")
inp = input("enter 1, 2 or 3: ")
if inp == "1":
    train()
elif inp == "2":
    eval1()
elif inp == "3":
    eval2()
else:
    print("Invalid option.\n")