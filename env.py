import numpy as np

class ArmEnv(object):
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 9 # TO DO
    action_dim = 3

    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2) and a3 start
        finger = np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)])*a3l + a2xy_ # a3 end
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400,
                 (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 400,
                 (self.goal['y'] - a2xy_[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist3[0]**2+dist3[1]**2)

        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, a2xy_/200, finger/200, dist1 +
                           dist2 + dist3, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2) and a3 start
        finger = np.array([np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)])*a3l + a2xy_ # a3 end
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400,
                 (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 400,
                 (self.goal['y'] - a2xy_[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 +
                           dist2, [1. if self.on_goal else 0.]))
        return s

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians
