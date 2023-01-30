import numpy as np
import pyglet
from pyglet import shapes


class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 13
    action_dim = 3

    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
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
        a1xy = np.array([300., 0.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * \
            a2l + a1xy_  # a2 end (x2, y2) and a3 start
        finger = np.array(
            [np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)])*a3l + a2xy_  # a3 end
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 600,
                 (self.goal['y'] - a1xy_[1]) / 600]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 600,
                 (self.goal['y'] - a2xy_[1]) / 600]
        dist3 = [(self.goal['x'] - finger[0]) / 600,
                 (self.goal['y'] - finger[1]) / 600]
        r = -np.sqrt(dist3[0]**2+dist3[1]**2)

        if a1xy_[1] < 0 or a2xy_[1] < 0 or finger[1] < 0:
            if a1xy_[1] < 0:
                r -= 1
            if a2xy_[1] < 0:
                r -= 1
            if finger[1] < 0:
                r -= 1
            self.on_goal = 0
        else:
            if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
                if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                    r += 1.
                    self.on_goal += 1
                    if self.on_goal > 50:
                        # r += (200 - self.on_goal)
                        done = True
            else:
                self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, a2xy_/200, finger/200, dist1 +
                           dist2 + dist3, [1. if self.on_goal else 0.]))

        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*600.
        self.goal['y'] = np.random.rand()*300.
        # test that goal is within reach of arm:
        while np.sqrt((self.goal['x']-300)**2 + self.goal['y']**2) > 300.:
            self.goal['x'] = np.random.rand()*600.
            self.goal['y'] = np.random.rand()*300.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)
        self.on_goal = 0
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([300., 0.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + \
            a1xy  # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * \
            a2l + a1xy_  # a2 end (x2, y2) and a3 start
        finger = np.array(
            [np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)])*a3l + a2xy_  # a3 end
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 600,
                 (self.goal['y'] - a1xy_[1]) / 600]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 600,
                 (self.goal['y'] - a2xy_[1]) / 600]
        dist3 = [(self.goal['x'] - finger[0]) / 600,
                 (self.goal['y'] - finger[1]) / 600]
        # state
        s = np.concatenate((a1xy_/200, a2xy_/200, finger/200, dist1 +
                           dist2 + dist3, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=600, height=300,
                                     resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([300, 0])
        self.mouse = True

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(4, pyglet.gl.GL_QUADS, None,    # 4 corners
                                   ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                                            goal['x'] - goal['l'] / \
                                            2, goal['y'] + goal['l'] / 2,
                                            goal['x'] + goal['l'] / \
                                            2, goal['y'] + goal['l'] / 2,
                                            goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
                                #    ('c3B', (255, 255, 255) * 4))    # color
                                   ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [250, 250,                # location
                                            250, 300,
                                            260, 300,
                                            260, 250]),
                                   ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [100, 150,              # location
                                            100, 160,
                                            200, 160,
                                            200, 150]),
                                   ('c3B', (249, 86, 86) * 4,))
        self.arm3 = self.batch.add(4, pyglet.gl.GL_QUADS, None,
                                   ('v2f', [100, 150,              # location
                                            100, 160,
                                            200, 160,
                                            200, 150]),
                                   ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l'] /
            2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l'] /
            2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l'] /
            2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l, a3l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r, a3r) = self.arm_info['r']  # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)])*a1l + \
            a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * \
            a2l + a1xy_  # a2 end (x2, y2)
        a3xy_ = np.array(
            [np.cos(a1r+a2r+a3r), np.sin(a1r+a2r+a3r)])*a3l + a2xy_  # a3 end

        a1tr, a2tr, a3tr = np.pi / 2 - self.arm_info['r'][0], \
            np.pi / 2 - self.arm_info['r'][0] + self.arm_info['r'][1], \
            np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        xy21_ = a2xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc
        xy22_ = a2xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy31 = a3xy_ + np.array([np.cos(a3tr), -np.sin(a3tr)]) * self.bar_thc
        xy32 = a3xy_ + np.array([-np.cos(a3tr), np.sin(a3tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))
        self.arm3.vertices = np.concatenate((xy21_, xy22_, xy31, xy32))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        if self.mouse:
            self.goal_info['x'] = x
            self.goal_info['y'] = y
        # print(self.goal_info)

    def draw_circles(self, filename):
        f = open(filename, 'r')
        inp = f.readline()
        self.circles = []
        while inp != "":
            inp = inp.split()
            circle = shapes.Circle(x=float(inp[0]), y=float(inp[1]), radius=5, color=(0, 0, 0), batch=self.batch)
            self.circles.append(circle)
            inp = f.readline()
        f.close()


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
