from mass_spring_robot_config import robots
import random
import sys
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os
import torch


max_steps = 4096
n_objects, n_springs = 0, 0
x = torch.zeros(max_steps, requires_grad=True)
v = torch.zeros(max_steps, requires_grad=True)
v_inc = torch.zeros(max_steps, requires_grad=True)
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()


def setup_robot(objects, springs):
    global n_objects, n_springs
    n_objects = len(objects)
    n_springs = len(springs)

    print('n_objects=', n_objects, '   n_springs=', n_springs)

    for i in range(n_objects):
        x[0, i] = objects[i]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]

@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])


def clear():
    clear_states()

def optimize(toi, visualize):
    global use_toi
    use_toi = toi
    # for i in range(n_hidden):
    #     for j in range(n_input_states()):
    #         weights1[i, j] = np.random.randn() * math.sqrt(
    #             2 / (n_hidden + n_input_states())) * 2

    # for i in range(n_springs):
    #     for j in range(n_hidden):
    #         # TODO: n_springs should be n_actuators
    #         weights2[i, j] = np.random.randn() * math.sqrt(
    #             2 / (n_hidden + n_springs)) * 3

    losses = []
    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(100):
        clear()
        # with ti.Tape(loss) automatically clears all gradients
        with ti.Tape(loss):
            forward(visualize=visualize)

        print('Iter=', iter, 'Loss=', loss[None])

        # total_norm_sqr = 0
        # for i in range(n_hidden):
        #     for j in range(n_input_states()):
        #         total_norm_sqr += weights1.grad[i, j]**2
        #     total_norm_sqr += bias1.grad[i]**2
        #
        # for i in range(n_springs):
        #     for j in range(n_hidden):
        #         total_norm_sqr += weights2.grad[i, j]**2
        #     total_norm_sqr += bias2.grad[i]**2
        #
        # print(total_norm_sqr)
        #
        # # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        # gradient_clip = 0.2
        # scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        # for i in range(n_hidden):
        #     for j in range(n_input_states()):
        #         weights1[i, j] -= scale * weights1.grad[i, j]
        #     bias1[i] -= scale * bias1.grad[i]
        #
        # for i in range(n_springs):
        #     for j in range(n_hidden):
        #         weights2[i, j] -= scale * weights2.grad[i, j]
        #     bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

    return losses

robot_id = 0
if len(sys.argv) != 3:
    print(
        "Usage: python3 mass_spring.py [robot_id=0, 1, 2, ...] [task=train/plot]"
    )
    exit(-1)
else:
    robot_id = int(sys.argv[1])
    task = sys.argv[2]


def main():
    setup_robot(*robots[robot_id]())

    if task == 'plot':
        ret = {}
        for toi in [False, True]:
            ret[toi] = []
            for i in range(5):
                losses = optimize(toi=toi, visualize=False)
                # losses = gaussian_filter(losses, sigma=3)
                plt.plot(losses, 'g' if toi else 'r')
                ret[toi].append(losses)

        import pickle
        pickle.dump(ret, open('losses.pkl', 'wb'))
        print("Losses saved to losses.pkl")
    else:
        optimize(toi=True, visualize=True)
        clear()
        forward('final{}'.format(robot_id))

if __name__ == '__main__':
    main()