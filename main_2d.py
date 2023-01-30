"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env_2d import ArmEnv
from rl import DDPG
import time

MAX_EPISODES = 5000
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
    f1 = open("./accuracy.txt", 'w')
    f2 = open("./rewards.txt", 'w')
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
                f1.write("%f\n" % acc)
                f2.write("%f\n" % ep_r)
                print('Ep: %i | %s | ep_r: %.1f | step: %i | acc:' % (i, '----' if not done else 'done', ep_r, j), acc)
			    # print(a)
                break
    f1.close()
    f2.close()
    rl.save()


def eval(mouse = True, filename = None):
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    if filename != None: 
        env.viewer.draw_circles(filename)
        f = open(filename, 'r')
    s = env.reset()
    episodes = 0
    completed = 0
    while True:
        steps = 0
        done = False
        episodes += 1
        r = 0

        if not mouse:
            env.viewer.mouse = False
            if filename == None:
                inp = input()
            else:
                inp = f.readline()
                if inp == "":
                    f.close()
                    print("Accuracy during trajectory tracking is", completed/episodes)
                    break
            # print(inp)
            inp = inp.split()
            env.goal['x'] = float(inp[0])
            env.goal['y'] = float(inp[1])
            env.viewer.goal_info = env.goal
        
        while not done and steps < 20:
            steps += 1
            env.render()
            time.sleep(0.01)
            a = rl.choose_action(s)
            s, r, done = env.step(a)
        
        if r > 0:
            completed += 1


print("Options:\n1. Train\n2. Simulate using cursor\n3. Simulate using given points\n4. Simulate using a file with coordinates.")
inp = input("enter 1, 2, 3 or 4: ")
if inp == "1":
    train()
elif inp == "2":
    eval()
elif inp == "3":
    print("Enter x and y coordinates separated by space and press enter.")
    eval(False)
elif inp == "4":
    filename = input("enter filename: ")
    eval(False, filename)
else:
    print("Invalid option.\n")