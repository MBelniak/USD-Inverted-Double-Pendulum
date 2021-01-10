# USD-Inverted-Double-Pendulum
Double deep QLearning and A3C algorithms on InvertedDoublePendulum-v2 from OpenAI Gym.

![Model preformance with A3C](videos/GIF_18-12-2020_17-15-36.gif)

### Requirements:
Mujoco (1.50 for Windows) from [https://www.roboti.us/index.html](https://www.roboti.us/index.html)

PyTorch > 1.0

imageio-ffmpeg for recording videos of simulation

### Usage:

Run it from `src` directory:
```
python main.py
```
for training with default parameters. Default algorithm is DDQN.

##### Available parameters:
  `--algorithm {A3C,DDQN}`   Algorithm to use.

  `--load_file LOAD_FILE` Custom filename from which to load models before rendering.<br>
  By default, trained models are saved to file `<algorithm>--<episodes>-<threads>-<discount>-<step_max>-<actor_lr>-<critic_lr>`<br>
  For example: `A3C--1000000-5-0_99-5-0_001-0_001`

  `--threads THREADS`    Number of threads for A3C.

  `--episodes EPISODES`   Number of episodes for training process.

  `--discount DISCOUNT`   Discount rate.

  `--step_max STEP_MAX`  Max actor's steps before update of global model in A3C.

  `--actor_lr ACTOR_LR`  Actor's learning rate.

  `--critic_lr CRITIC_LR` Critic's learning rate.

  `--eval_repeats EVAL_REPEATS` Number of evaluation runs in one performance evaluation. Set to 0 to disable evaluation during training.

  `-no_log`  Disable logging during training.

  `-render`  Render environment. Before rendering, there must exist a model
  saved in a file which name is generated based on parameters or explicitly provided.