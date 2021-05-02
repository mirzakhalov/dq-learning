

from datetime import datetime
from collections import deque


import argparse
import numpy as np
import tensorflow as tf
import imageio
import gym

from agent import PongNetwork, Environment
from utils import ReplayMemory

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help="Path to the model checkpoint")

    args = parser.parse_args()
    return args

def get_action(net, state, exploration_rate):
    recent_state = tf.expand_dims(state, axis=0)
    if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < exploration_rate:
        action = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    else:
        q_value = net(tf.cast(recent_state, tf.float32))
        action = tf.cast(tf.squeeze(tf.math.argmax(q_value, axis=1)), dtype=tf.int32)
    return action


if __name__ == "__main__":

    args = args_parse()

    trial = 10
    
    if args.model_dir:
        print(args.model_dir)
        loaded_ckpt = tf.train.latest_checkpoint(args.model_dir)
        net = PongNetwork(6, 4)

        print(loaded_ckpt)
        net.load_weights(loaded_ckpt)
    
    frame_set = []
    reward_set = []
    test_env = Environment("PongNoFrameskip-v4", train=False)

    for i in range(trial):

        state = test_env.reset()
        frames = []
        test_step = 0
        test_reward = 0
        done = False
        test_memory = ReplayMemory(10000, verbose=False)

        while not done:

            frames.append(test_env.render())

            action = get_action(net, tf.constant(state, tf.float32), tf.constant(0.0, tf.float32))
    
            next_state, reward, done, info = test_env.step(action)
            test_reward += reward

            test_memory.push(state, action, reward, next_state, done)
            state = next_state

            test_step += 1

            if done and (info["ale.lives"] != 0):
                test_env.reset()
                test_step = 0
                done = False

        reward_set.append(test_reward)
        frame_set.append(frames)

    best_score = np.max(reward_set)
    print("Best score of current network ({} trials): {}".format(trial, best_score))
    best_score_ind = np.argmax(reward_set)
    imageio.mimsave("final_result.gif", frame_set[best_score_ind], fps=15)

