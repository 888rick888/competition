import numpy as np
import random
from collections import deque
import time
import pandas as pd
import scipy.signal

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, BatchNormalization, Add, Concatenate, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications import EfficientNetB0 as EfficientNet
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
import tensorflow_probability as tfp
from tensorflow.python.keras.utils.vis_utils import plot_model

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpus = tf.config.experimental.list_physical_devices('GPU')#获取GPU列表
# print('----gpus---',gpus)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)])

from olympics_engine.train.train_ppo import RENDER
from olympics_engine.scenario import table_hockey, football, wrestling, Running_competition
from olympics_engine.agent import *
from olympics_engine.generator import create_scenario


import matplotlib.pyplot as plt
import wandb
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # 隐藏warning

from BUFFER import Memory
from Noise import ActionNoise, AdaptiveParamNoiseSpec

RENDER = 1

RECORD_LOSS = 0
TRAIN = 1
PLOT = 1
PER = 0

STATE_SIZE = 1600
ACTION_DIM = 2

BUFF_SIZE = 1000000
TRAIN_EPISODES = 10000

BATCH_SIZE = 128
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.95
TAU = 0.005
TRAIN_ACTOR_EACH_C = 3 # each X+1 train
NOISE_EXPLORE_STD = 0.2
NOISE_POLICY_STD = 0.4
REGULARIZER_L2 = 0.001

TEST_EPISODES = 10
MAX_STEPS = 10000

EPSILON = 0.999
# EPSILON = 0.001
EPSILON_DECAY = 0.999

RANDOM_ACTION = 1
PLOT = 1

wandb.init(project="olimpics", entity="rickkkkk", reinit=True, name="td3_first")
wandb.config.hyper_patamter = {
    "State_size": STATE_SIZE,
    "learning_rate_Actor": LR_A,
    "learning_rate_Critic": LR_C,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "Tau":TAU,
    "Train_actor_each_critic":TRAIN_ACTOR_EACH_C,
    "Explore_noise_std":NOISE_EXPLORE_STD,
    "Policy_noise_std" : NOISE_POLICY_STD,
    "Regularizer_L2": REGULARIZER_L2,
    "PER_IS": PER,
}

# def my_handler(signum, frame):
#     global stop
#     stop = True
#     print("============ S T O P ============")
class Agent(object):
    def __init__(self):
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.epsilon_decay = EPSILON_DECAY
        self.tau = TAU
        # self.memory = deque(maxlen=400000)
        self.batch_size = BATCH_SIZE
        self.buff = Memory(BUFF_SIZE, PER)
        self.noise = ActionNoise(mu=0, sigma=1)
        self.train_count = 0
        self.action_bound_force = 200
        self.action_bound_angle = 30

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
 
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_state_input_2, self.critic_action_input_2, self.critic_model_2 = self.create_critic_model()
        _, _, self.target_critic_model_2 = self.create_critic_model()

        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)

    def create_actor_model(self):
        steer_input = Input(shape=(ACTION_DIM))
        state_input = Input(shape=(STATE_SIZE))
        h1 = Dense(6400,activation='elu')(state_input)
        h2 = Dense(3200,activation='elu')(h1)
        h3 = Dense(800,activation='elu')(h2)
        h4 = Dense(256,activation='elu')(h3)
        state_output = Dense(128,activation='elu')(h4)
        steer_output = Dense(32, activation='elu')(steer_input)
        merge1 = Concatenate()([state_output,steer_output])
        # h11 = Dense(512,activation='elu')(merge1)
        h22 = Dense(256,activation='elu')(merge1)
        action_out = Dense(ACTION_DIM,activation='tanh')(h22)
        # action_out = Dense(ACTION_DIM)(h22)
        model = Model(inputs=[state_input, steer_input], outputs=action_out)
        return state_input, model
    
    def create_critic_model(self):
        steer_input_c = Input(shape=(ACTION_DIM))
        state_input_c = Input(shape=(STATE_SIZE))
        h1_c = Dense(6400,activation='elu')(state_input_c)
        h2_c = Dense(3200,activation='elu')(h1_c)
        h3_c = Dense(800,activation='elu')(h2_c)
        h4_c = Dense(256,activation='elu')(h3_c)
        state_output_c = Dense(128,activation='elu')(h4_c)
        steer_output_c = Dense(32, activation='elu')(steer_input_c)
        merge1_c = Concatenate()([state_output_c,steer_output_c])
        # h11_c = Dense(512,activation='elu')(merge1_c)
        h22_c = Dense(256,activation='elu')(merge1_c)
        # action_out = Dense(ACTION_DIM)(h22)
        action_out_c = Dense(ACTION_DIM,activation='tanh')(h22_c)
        model_c = Model(inputs=[state_input_c, steer_input_c], outputs=action_out_c)
        return state_input_c, steer_input_c, model_c

    def remember(self,s_t,action,reward,s_t1,done):
        experiences = (s_t,action,reward,s_t1,done)
        self.buff.store(experiences)
        # self.memory.append(experiences)


    def train(self):
        if self.buff.num < self.batch_size: 
            return

        self.train_count +=1
        tree_idx, samples, self.ISWeights = self.buff.sample(self.batch_size)

        self.samples = samples
        # print(np.shape(samples), samples)
        self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones = self.stack_samples(self.samples)
        # self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones = self.s_ts.astype(np.float32), self.actions.astype(np.float32), self.rewards.astype(np.float32), self.s_ts1.astype(np.float32), self.dones.astype(np.float32)
        target_actions = self.target_actor_model([self.s_ts1, self.actions], training=True)
        target_actions = tf.convert_to_tensor(self.noise.add_noise(target_actions, std=NOISE_POLICY_STD))
        q_value, q_target = self.train_net(self.s_ts, self.actions, self.rewards, self.s_ts1, self.dones, target_actions)
        if self.train_count >= TRAIN_ACTOR_EACH_C:
            self.train_count = 0
            self.train_actor(self.s_ts, self.actions)

        if PER:
            err_batch = self.update_err(np.asarray(q_value), np.asarray(q_target))
            self.buff.batch_update(tree_idx, err_batch)

    @tf.function
    def train_actor(self,s_ts, actions):
        with tf.GradientTape() as tape:
            actions_out = self.actor_model([s_ts, actions], training=True)
            critic_value_a = self.critic_model([s_ts, actions_out], training=True)
            # actor_loss = - 0.1 * tf.math.reduce_mean(critic_value_a)
            actor_loss = - tf.math.reduce_mean(critic_value_a)
            # actor_loss += self.action_l2 * tf.math.reduce_mean(tf.square(critic_value_a / self.max_u))

        # tf.print("=====The actor_loss is ", actor_loss, "=====")
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))
        if RECORD_LOSS:
            wandb.log({"Actor_loss": actor_loss})

    # @tf.function
    def train_net(self,s_ts, actions, rewards, s_ts1, dones, target_actions):
        target_value = self.target_critic_model([s_ts1, target_actions], training=True)
        target_value_2 = self.target_critic_model_2([s_ts1, target_actions], training=True)

        rewards = np.reshape(rewards, (BATCH_SIZE, 1))
        rewards = np.concatenate([rewards, rewards], axis=1)
        dones = np.reshape(dones, (BATCH_SIZE, 1))
        dones = np.concatenate([dones, dones], axis=1)

        y = rewards + (1-dones) * self.gamma * tf.minimum(target_value, target_value_2)
        with tf.GradientTape() as tape:
            critic_value = self.critic_model([s_ts, actions], training=True)
            critic_loss = self.ISWeights * tf.math.reduce_mean( tf.math.square(y - critic_value))

        # tf.print("+++++ The critic loss is ", critic_loss, "+++++")
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            critic_value_2 = self.critic_model_2([s_ts, actions], training=True)
            critic_loss_2 = self.ISWeights * tf.math.reduce_mean( tf.math.square(y - critic_value_2))

        critic_grad_2 = tape.gradient(critic_loss_2, self.critic_model_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad_2, self.critic_model_2.trainable_variables))
        
        if RECORD_LOSS:
            wandb.log({"Critic1_loss": critic_loss, "Critic2_loss": critic_loss_2})
            wandb.log({"Critic1_value": critic_value, "Critic2_value": critic_value_2})
            wandb.log({"target_value_2": target_value_2, "target_value": target_value})
        return critic_value, y

    def update_target(self):
        # self.update_actor_target()
        # self.update_critic_target()
        self.update_target_each(self.target_actor_model.variables, self.actor_model.variables, self.tau)
        self.update_target_each(self.target_critic_model.variables, self.critic_model.variables, self.tau)
        self.update_target_each(self.target_critic_model_2.variables, self.critic_model_2.variables, self.tau)
    
    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)
    
    @tf.function
    def update_target_each(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
 
    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

        critic_model_2_weights = self.critic_model_2.get_weights()
        critic_target_2_weights = self.target_critic_model_2.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_2_weights[i] = critic_model_2_weights[i]*self.tau + critic_target_2_weights[i]*(1-self.tau)
        self.target_critic_model_2.set_weights(critic_target_2_weights)
 
    def act(self,s_t, control=None, get_action=np.ones(2, )):
        s_t = s_t[np.newaxis, :].astype(np.float32)
        get_action = get_action[np.newaxis, :].astype(np.float32)
        # print("========= Control is ===", control.steer)
        a_t = np.zeros((ACTION_DIM))
        if self.epsilon > 0.001:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            # control = self.ue4.agent_action()
            a_t[0] = np.random.uniform(-100, 200)
            a_t[1] = np.random.uniform(-self.action_bound_angle, self.action_bound_angle)
            # a_t[0][0] = control.steer
            # a_t[0][1] = control.throttle
            return a_t
        a_predict = self.actor_model([s_t, get_action])
        # action_out = np.clip(np.squeeze(self.noise.add_noise(a_predict, std=NOISE_EXPLORE_STD)), [-100, -self.action_bound_angle], [200, self.action_bound_angle])
        action_out = np.squeeze(self.noise.add_noise(a_predict, std=NOISE_EXPLORE_STD))
        print("the origin action is ", a_predict, "the action_out is ", action_out)
        return action_out

    def stack_samples(self, samples):       #maybe
        s_ts = np.array([e[0] for e in samples], dtype='float32')
        actions = np.array([e[1] for e in samples], dtype='float32')
        rewards = np.array([e[2] for e in samples], dtype='float32')
        s_ts1 = np.array([e[3] for e in samples], dtype='float32')
        dones = np.array([e[4] for e in samples], dtype='float32')

        s_ts = tf.convert_to_tensor(s_ts)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        s_ts1 = tf.convert_to_tensor(s_ts1)
        dones = tf.convert_to_tensor(dones)
        return s_ts, actions, rewards, s_ts1, dones

    def update_err(self,q_value, q_target):       #maybe 
        err_batch = np.zeros((len(q_target), 1))
        for i in range(len(q_target)):
            err = 0
            for o in range(len(q_target[0])):
                err += np.abs(q_value[i][o] - q_target[i][o])
            err_batch[i][0] = err / len(q_target[0])
        return err_batch
    
    def save_model(self):
        try:
            self.target_actor_model.save(f"carla_ddpg_actor.h5", include_optimizer=False)
            self.target_critic_model.save(f"carla_ddpg_critic.h5", include_optimizer=False)
            self.target_critic_model_2.save(f"carla_ddpg_critic_2.h5", include_optimizer=False)
            print("------------------------------- save model ----------------------------------")
        except:
            print('---------------------------Can not save model----------------------------------')

    def load_model(self):
        try:
            self.actor_model=load_model(f"carla_ddpg_actor.h5")
            self.critic_model=load_model(f"carla_ddpg_critic.h5")
            self.critic_model_2=load_model(f"carla_ddpg_critic_2.h5")
            print("------------------------------- load model ----------------------------------")
        except:
            print('---------------------------Can not load model----------------------------------')

    def target_setweight(self):
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.target_critic_model_2.set_weights(self.critic_model_2.get_weights())
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        
    def plot(episodes, rewards):
        if PLOT:
            # clear_output(True)
            plt.figure(figsize=(20, 5))
            plt.title('episode. reward')
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Each Episode Reward')
            plt.savefig('reward_episode.png')
            plt.close('all')
            # plt.show()
        else: pass

if __name__ == "__main__":
    num_agents = 2
    ctrl_agent_index = 0 
    # signal.signal(signal.SIGINT, my_handler)    #stop signal
    # stop = False
    rewards = []
    final_step = []
    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    agent = Agent()
    Gamemap = create_scenario('wrestling')
    env = wrestling(Gamemap)
    env.max_step = 400

    agent.load_model()
    agent.target_setweight()
    opponent_agent = random_agent()     #we use random opponent agent here

    # adam_a = Adam(learning_rate=LR_A)
    # adam_c = Adam(learning_rate=LR_C)
    # agent.critic_model.compile(loss="mse",optimizer=adam_c, metrics=["acc"])
    # agent.actor_old_model.compile(loss="mse",optimizer=adam_a, metrics=["acc"])
    # agent.actor_model.compile(loss="mse",optimizer=adam_a, metrics=["acc"])

    t0 = time.time()
    if TRAIN :
        for e in range(TRAIN_EPISODES):
            state = env.reset()
            epoch_reward = 0
            step = 0
            Gt = 0
            train_count = 0
            if RENDER :
                env.render()

            if isinstance(state[ctrl_agent_index], type({})):
                obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
            else:
                obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index].flatten(), env.agent_list[ctrl_agent_index].energy
                obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

            while True:
                step += 1

                action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
                # action_opponent = [0,0]  #here we assume the opponent is not moving in the demo

                action_ctrl= agent.act(obs_ctrl_agent) #action_ctrl[0] is force , action_ctrl[1] is angle
                action_ctrl_t = np.array(action_ctrl)
                action_ctrl_t[0] = np.clip(action_ctrl_t[0] * 200, -100, 200)
                action_ctrl_t[1] = np.clip(action_ctrl_t[1] * 30, -agent.action_bound_angle, agent.action_bound_angle)
                action = [action_opponent, action_ctrl_t] if ctrl_agent_index == 1 else [action_ctrl_t, action_opponent]

                print("action is ", action_ctrl)
                next_state, reward, done, _ = env.step(action)
                # reward[0] += 0.1
                
                if isinstance(next_state[ctrl_agent_index], type({})):
                    next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                    next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
                else:
                    next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
                    next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy
                next_obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
                if not done:
                    post_reward = [0, 0]
                else:
                    if reward[0] != reward[1]:
                        post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                    else:
                        post_reward=[0, 0]

                agent.remember(obs_ctrl_agent, action_ctrl, reward[0], next_obs_ctrl_agent, done)
                epoch_reward +=reward[0]

                obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
                obs_ctrl_agent, energy_ctrl_agent = next_obs_ctrl_agent, next_energy_ctrl_agent

                agent.train()

                if step % 3 ==0:
                    agent.update_target()

                if RENDER:
                    env.render()
                Gt += reward[ctrl_agent_index] if done else -1

                if done:
                    win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                    win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                    record_win.append(win_is)
                    record_win_op.append(win_is_op)
                    print("Episode: ", e, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                        "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                        '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)
                    break

                # try:
                #     if stop:
                #         time.sleep(2)
                #         raise ValueError ('stop from keyboard')
                # except Exception as e:
                #     print(str(e))
                #     raise ValueError ('stop from keyboard')

            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    e + 1, TRAIN_EPISODES, epoch_reward, time.time() - t0))
    
                # print("episode:{} step:{} predict_action:{} action:{} reward:{}".format(e,step,a_predict,action,reward))
            if e == 0:
                rewards.append(epoch_reward)
            else:
                rewards.append(rewards[-1] * 0.9 + epoch_reward * 0.1)
            final_step.append(step)

            # wandb.log({"rewards": rewards[-1], "final_step": step})
            # wandb.log({"final_step": step})

            # if e%2000 == 0 and e is not 0:
            #     plot(final_step,rewards)
            
            if e%500 == 0 and e is not 0:
                agent.save_model()
    
    else :
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                state, reward, done, _ = env.step(agent.get_action(state, greedy=True))
                episode_reward += reward
                if done:
                    break

                try:
                    if stop:
                        env.des()
                        time.sleep(2)
                        raise ValueError ('stop from keyboard')
                except Exception as e:
                    print(str(e))
                    raise ValueError ('stop from keyboard')
            env.des()
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))