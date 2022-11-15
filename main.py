"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env import ArmEnv
from rl import DDPG

MAX_EPISODES = 10000
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
                    acc = 0.1 + 0.9*acc
                else:
                    acc = 0.9*acc
                print('Ep: %i | %s | ep_r: %.1f | step: %i | acc:' % (i, '---' if not done else 'done', ep_r, j), acc)
			    # print(a)
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)

ON_TRAIN = "1" == input("enter 1 for train, 0 for simulate: ")
if ON_TRAIN:
    train()
else:
    eval()