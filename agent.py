import numpy as np
import tensorflow as tf
import imageio
import gym
from datetime import datetime
from collections import deque

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda

from utils import ReplayMemory, make_atari, wrap_deepmind, Environment


class PongNetwork(Model):

    def __init__(self, num_actions: int, agent_history_length: int):
        super(PongNetwork, self).__init__()
        self.normalize = Lambda(lambda x: x / 255.0)
        self.conv1 = Conv2D(filters=16, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu", input_shape=(None, 84, 84, agent_history_length))
        self.conv2 = Conv2D(filters=32, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv4 = Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")

        self.flatten = Flatten()
        
        self.dense1 = Dense(512, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')
        self.dense2 = Dense(256, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation='relu')
        self.dense3 = Dense(num_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="linear")

    # same as forward()
    def call(self, x):
        # pass the input through all the layers
        out = self.dense3(
                    self.dense2(
                        self.dense1(
                            self.flatten(
                                self.conv4(
                                    self.conv3(
                                        self.conv2(
                                            self.conv1(
                                                self.normalize(x)
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )

        return out





# helpful repos on this section: https://github.com/jihoonerd/Deep-Reinforcement-Learning-with-Double-Q-learning 
class Agent:
    
    def __init__(self, args):

        # which environment to load from the opencv database
        self.env_id = "PongNoFrameskip-v4"
        # create the environment
        self.env = Environment(self.env_id)

        # part of the q-value formula
        self.discount_factor = 0.99
        self.batch_size = 64
        # how often to update the network (backpropogation)
        self.update_frequency = 4
        # often synchronize with the target  network
        self.target_network_update_freq = 1000

        # keeps track of the frames for training, and retrieves them in batches 
        self.agent_history_length = 4
        self.memory = ReplayMemory(capacity=10000, batch_size=self.batch_size)

        # two neural networks. One for main and one for target
        self.main_network = PongNetwork(num_actions=self.env.get_action_space_size(), agent_history_length=self.agent_history_length)
        self.target_network = PongNetwork(num_actions=self.env.get_action_space_size(), agent_history_length=self.agent_history_length)
        
        # adam optimizer. just a standard procedure
        self.optimizer = Adam(learning_rate=1e-4, epsilon=1e-6)
        # we start with a high exploration rate then slowly decrease it
        self.init_explr = 1.0
        self.final_explr = 0.1
        self.final_explr_frame = 1000000
        self.replay_start_size = 10000

        # metrics for the loss 
        self.loss = tf.keras.losses.Huber()
        # this will be the mean of 100 last rewards
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        # comes from the q loss below
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")

        # what is the max number of frames to train. probably won't reach here.
        self.training_frames = int(1e7)

        # path to save the checkpoints, logs and the weights
        self.checkpoint_path = "./checkpoints/" + args.run_name
        self.tensorboard_writer = tf.summary.create_file_writer(self.checkpoint_path + "/runs/")
        self.print_log_interval = 10
        self.save_weight_interval = 10
        self.env.reset()
           

     # calculate the network loss on the replay buffer (Q-learning)
    def update_main_q_network(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
       
        with tf.GradientTape() as tape:
            ## THIS IS WHERE THE MAGIC HAPPENS!
            ## L = Q(s, a) - (r + discount_factor* Max Q(s’, a))
            next_state_q = self.target_network(next_state_batch)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = reward_batch + self.discount_factor * next_state_max_q * (1.0 - tf.cast(terminal_batch, tf.float32))
            main_q = tf.reduce_sum(self.main_network(state_batch) * tf.one_hot(action_batch, self.env.get_action_space_size(), 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss

    
     # calculate the network loss on the replay buffer (Double Q-learning)
    def update_main_dq_network(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        
        with tf.GradientTape() as tape:
            # THIS IS WHERE THE MAGIC HAPPENS!
            ## here we maintain two Q values: one to maximize the reward in the next state and one to update current state
            q_online = self.main_network(next_state_batch)  # Use q values from online network
            action_q_online = tf.math.argmax(q_online, axis=1)  # optimal actions from the q_online
            q_target = self.target_network(next_state_batch)  #  q values from target netowkr
            ddqn_q = tf.reduce_sum(q_target * tf.one_hot(action_q_online, self.env.get_action_space_size(), 1.0, 0.0), axis=1)
            expected_q = reward_batch + self.discount_factor * ddqn_q * (1.0 - tf.cast(terminal_batch, tf.float32))  # Corresponds to equation (4) in ddqn paper
            main_q = tf.reduce_sum(self.main_network(state_batch) * tf.one_hot(action_batch, self.env.get_action_space_size(), 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss



    # get the next action index based on the state (84,84,4) and exploration rate
    def get_action(self, state, exploration_rate):
        recent_state = tf.expand_dims(state, axis=0)
        if tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < exploration_rate:
            action = tf.random.uniform((), minval=0, maxval=self.env.get_action_space_size(), dtype=tf.int32)
        else:
            q_value = self.main_network(tf.cast(recent_state, tf.float32))
            action = tf.cast(tf.squeeze(tf.math.argmax(q_value, axis=1)), dtype=tf.int32)
        return action
        
    
    # get the epsilon value for the current based. Similar to https://openai.com/blog/openai-baselines-dqn/
    def get_eps(self, current_step, terminal_eps=0.01, terminal_frame_factor=25):
    
        terminal_eps_frame = self.final_explr_frame * terminal_frame_factor

        if current_step < self.replay_start_size:
            eps = self.init_explr
        elif self.replay_start_size <= current_step and current_step < self.final_explr_frame:
            eps = (self.final_explr - self.init_explr) / (self.final_explr_frame - self.replay_start_size) * (current_step - self.replay_start_size) + self.init_explr
        elif self.final_explr_frame <= current_step and current_step < terminal_eps_frame:
            eps = (terminal_eps - self.final_explr) / (terminal_eps_frame - self.final_explr_frame) * (current_step - self.final_explr_frame) + self.final_explr
        else:
            eps = terminal_eps
        return eps
    
        
    # copy over the weights between the main and target network to synchronize
    def update_target_network(self):
        main_vars = self.main_network.trainable_variables
        target_vars = self.target_network.trainable_variables
        for main_var, target_var in zip(main_vars, target_vars):
            target_var.assign(main_var)

    def train(self, algorithm='q'):
    
        total_step = 0
        episode = 0
        latest_mean_score = -99.99
        latest_100_score = deque(maxlen=100)
        # this is kinda arbitrary but looks like the best bot reach 20 when they are done training in this game
        max_reward = 20.0

        # train until the mean reward reaches 20
        while latest_mean_score < max_reward:
            
            # reset the variable for the upcoming episode
            state = self.env.reset()
            episode_step = 0
            episode_score = 0.0
            done = False


            while not done:
                # while the episode is not done, calculate the epsilon and get the next action
                eps = self.get_eps(tf.constant(total_step, tf.float32))
                action = self.get_action(tf.constant(state), tf.constant(eps, tf.float32))
            
                next_state, reward, done, info = self.env.step(action)
                episode_score += reward

                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                # update the netwrok
                if (total_step % self.update_frequency == 0) and (total_step > self.replay_start_size):
                    indices = self.memory.get_minibatch_indices()
                    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.generate_minibatch_samples(indices)
                    if algorithm == 'q':
                        self.update_main_q_network(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                    else:
                        self.update_main_dq_network(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

                if (total_step % self.target_network_update_freq == 0) and (total_step > self.replay_start_size):
                    loss = self.update_target_network()
                
                total_step += 1
                episode_step += 1

                if done:
                    latest_100_score.append(episode_score)
                    self.write_summary(episode, latest_100_score, episode_score, total_step, eps)
                    episode += 1

                    if episode % self.print_log_interval == 0:
                        print("Episode: ", episode)
                        print("Latest 100 avg: {:.4f}".format(np.mean(latest_100_score)))
                        print("Progress: {} / {} ( {:.2f} % )".format(total_step, self.training_frames, 
                        np.round(total_step / self.training_frames, 3) * 100))
                        latest_mean_score = np.mean(latest_100_score)

                    if episode % self.save_weight_interval == 0:
                        print("Saving weights...")
                        self.main_network.save_weights(self.checkpoint_path + "/weights/episode_{}".format(episode))


    # write the summaries back to the tensorboard
    def write_summary(self, episode, latest_100_score, episode_score, total_step, eps):

        with self.tensorboard_writer.as_default():
            tf.summary.scalar("Reward", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg rewards", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("Loss", self.loss_metric.result(), step=episode)
            tf.summary.scalar("Average Q", self.q_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)
            tf.summary.scalar("Epsilon", eps, step=episode)

        self.loss_metric.reset_states()
        self.q_metric.reset_states()




