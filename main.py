from env import ClutteredPushGrasp
from robot import UR5Robotiq85
from rl import DDPG
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MAX_EPISODES = 10000
MAX_EP_STEPS = 100


rl = DDPG(4, 31, [-0.1, 0.1], 5*MAX_EPISODES)


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
    f1.close()
    f2.close()
    rl.save()
    env.close()


def simulate(filename=None):
    rl.restore()
    env = ClutteredPushGrasp(UR5Robotiq85((0, 0.5, 0), (0, 0, 0)), vis=True)
    s = env.reset()
    if filename != None:
        f = open(filename, 'r')
    episodes = 0
    completed = 0
    while True:
        steps = 0
        done = False
        episodes += 1
        r = 0

        if filename == None:
            inp = input('\nEnter coordinates: ')
        else:
            inp = f.readline()
            if inp == "":
                f.close()
                print("Accuracy during trajectory tracking is", completed/episodes)
                break
        inp = inp.split()
        env.set_goal([float(inp[0]), float(inp[1]), float(inp[2])])

        while not done and steps < 100:
            steps += 1
            env.step_simulation()
            a = rl.choose_action(s)
            s, r, done = env.step(a)

        if r > 0:
            completed += 1
    env.close()


if __name__ == "__main__":
    print("Options:\n0 -> Train\n1 -> Simulate using given points\n2 -> Simulate using a file with coordinates")
    inp = input("enter 0, 1 or 2: ")
    if inp == "0":
        train()
    elif inp == "1":
        print("Enter x, y and z coordinates separated by space and press enter.")
        simulate()
    elif inp == "2":
        filename = input("Enter filename: ")
        simulate(filename)
    else:
        print("Invalid option.\n")
