# Abstract

Robotic manipulators play a crucial role in automating various industrial tasks that would otherwise require human intervention. In this project, we focus on developing a model-free deep reinforcement learning (RL) approach to control the trajectories of such robots. We have employed a 6-degree of freedom (DOF) robotic arm for this purpose and created suitable state and action representations. We utilized the Deep Deterministic Policy Gradient (DDPG) method, a highly adaptable deep RL technique, to train our RL agent. Initially, we applied the DDPG method to a 2D model of the robotic manipulator and achieved impressive accuracy and smoothness in tracking various 2D paths. After this success, we developed a 3D model using the Pybullet library and trained the same DDPG agent on it. The 3D model was trained using Python 3.8, using the TensorFlow and Pybullet libraries. Following the training, the 3D simulation demonstrated the ability to track complex 3D trajectories, such as helix, rectangle, crown, and more. We conducted extensive testing with numerous reward functions and neural network designs, ultimately arriving at an RL agent that enables our robotic arm to smoothly and swiftly track any given path. A significant advantage of our approach is its model-free nature, which contrasts with traditional model-based trajectory control methods that necessitate extensive calculations.

## Requirements

```
pybullet
pyglet
numpy
tensorflow
matplotlib
```

## Training

Use the following command to start training:
```
python3 train.py
```

## Testing / Simulation

For testing, select one of the trajectories from trajectories folder and pass the following command:
```
python3 eval.py trajectories/crown
```

## Plot Results

Plot actual and predicted trajectories in 3D space:

```
python3 plots/plot_3d.py trajectories/crown
python3 plots/plot_3d.py results/crown_pred_path
```

Plot training metrics:

```
python3 plots/plot_metrics.py results/accuracy.txt
```

Plot change in x, y and z between actual and tracked trajectory:

```
python3 plots/plot_xyz.py trajectories/crown results/crown_pred_path
```

Plot rd part of reward function:

```
python3 plots/plot_rd.py
```
