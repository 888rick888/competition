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

RANDOMSEED = 1
tf.random.set_seed(RANDOMSEED)
random.seed(RANDOMSEED)
np.random.seed(RANDOMSEED)

RENDER = 1

RECORD_LOSS = 0
TRAIN = 1
PLOT = 0
GAE = 1

STATE_SIZE = 1600
ACTION_DIM = 2

BATCH_SIZE = 128
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.98
TRAIN_EPISODES = 10000  # total number of episodes for training
MAX_STEPS = 400  # total number of steps for each episode
ACTOR_UPDATE_STEPS = 5  # actor update steps
CRITIC_UPDATE_STEPS = 5  # critic update steps
EPSILON = 0.2 # ppo-clip parameters
LAMBDA = 0.98

TEST_EPISODES = 10  # total number of episodes for testing
# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

if GAE:
    GAMMA = 0.98

wandb.init(project="olimpics", entity="rickkkkk", reinit=True, name="ppo_first")
wandb.config.hyper_patamter = {
    "State_size": STATE_SIZE,
    "learning_rate_Actor": LR_A,
    "learning_rate_Critic": LR_C,
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "Actor_uodate_steps":ACTOR_UPDATE_STEPS,
    "Critic_uodate_steps":CRITIC_UPDATE_STEPS,
    "lambda_GAE":LAMBDA,
}

# def my_handler(signum, frame):
#     global stop
#     stop = True
#     print("============ S T O P ============")

class Agent(object):
    def __init__(self):
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.action_bound_force = 200
        self.action_bound_angle = 30
        self.train_count = 0
        self.method = 'clip'
        self.lam = LAMBDA

        if self.method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif self.method == 'clip':
            self.epsilon = EPSILON

        self.actor_state_input, self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)
        self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)
        self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []

    def create_actor_model(self):
        state_input = Input(shape=(STATE_SIZE))
        h1 = Dense(6400,activation='elu')(state_input)
        h2 = Dense(3200,activation='elu')(h1)
        h3 = Dense(800,activation='elu')(h2)
        h4 = Dense(256,activation='elu')(h3)
        h5 = Dense(64,activation='elu')(h4)
        Steering_mean = Dense(ACTION_DIM)(h5)
        # Steering_mean = Dense(ACTION_DIM, activation='tanh')(h5)
        Steering_sigma = Dense(ACTION_DIM, activation='softplus')(h5)
        # force_mean = Dense(1, activation='tanh')(h5)
        # force_sigma = Dense(1, activation='softplus')(h5)
        # model = Model(inputs=state_input, outputs=[Steering_mean, Steering_sigma, force_mean, force_sigma])
        model = Model(inputs=state_input, outputs=[Steering_mean, Steering_sigma])
        return state_input, model
    
    def create_critic_model(self):
        state_input_c = Input(shape=(STATE_SIZE))
        h1 = Dense(6400,activation='elu')(state_input_c)
        h2 = Dense(3200,activation='elu')(h1)
        h3 = Dense(800,activation='elu')(h2)
        h4 = Dense(256,activation='elu')(h3)
        h5 = Dense(64,activation='elu')(h4)
        Steering = Dense(ACTION_DIM)(h5)
        # Steering = Dense(ACTION_DIM, activation='tanh')(h5)
        # force = Dense(1, activation='tanh')(h5)
        # model_c = Model(inputs=state_input_c, outputs=[Steering, force])
        model_c = Model(inputs=state_input_c, outputs=[Steering])
        return model_c

    # @tf.function
    def train_actor(self,state, action, adv, old_pi):
        with tf.GradientTape() as tape:
            mean, std = self.actor_model(state)
            pi = tfp.distributions.Normal(mean, std)

            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv))
        a_gard = tape.gradient(loss, self.actor_model.trainable_weights)
        # self.actor_opt.apply_gradients(zip(a_gard, self.actor_model.trainable_weights))
        self.actor_optimizer.apply_gradients(zip(a_gard, self.actor_model.trainable_weights))
        if self.method == 'kl_pen':
            return kl_mean

        if RECORD_LOSS:
            wandb.log({"Actor_loss": loss})

    def train_critic(self, reward, state):
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic_model(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic_model.trainable_weights)
        # self.critic_opt.apply_gradients(zip(grad, self.critic_model.trainable_weights))
        self.critic_optimizer.apply_gradients(zip(grad, self.critic_model.trainable_weights))
        
        if RECORD_LOSS:
            wandb.log({"Critic_loss": loss})

    def update(self):
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        mean, std = self.actor_model(s)
        pi = tfp.distributions.Normal(mean, std)
        adv = r - self.critic_model(s)
        if GAE:
            adv = self.discounted_cumulative_sums(adv, self.gamma * self.lam)
        
        print("==========  I am updating ~~~  ==========")
        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                self.train_actor(s, a, adv, pi)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()
    
    def get_action(self, state, greedy=False):
        state = state[np.newaxis, :].astype(np.float32)
        # b = time.time()
        mean, std = self.actor_model(state)
        # print('----------', time.time()-b)
        if greedy:
            action = mean[0]
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        action_out = np.clip(action, [-100, -self.action_bound_angle], [200, self.action_bound_angle])
        print("the origin action is ", action, "the action_out is ", action_out)
        return action_out

    def save_model(self):
        try:
            self.actor_model.save(f"carla_ppo_actor.h5", include_optimizer=False)
            # self.actor_old_model.save(f"carla_ppo_actor_old.h5", include_optimizer=False)
            self.critic_model.save(f"carla_ppo_critic.h5", include_optimizer=False)
            print("------------------------------- save model ----------------------------------")
        except:
            print('---------------------------Can not save model----------------------------------')


    def load_model(self):
        try:
            self.actor_model=load_model(f"carla_ppo_actor.h5")
            # self.actor_old_model=load_model(f"carla_ppo_actor_old.h5")
            self.critic_model=load_model(f"carla_ppo_critic.h5")
            print("------------------------------- load model ----------------------------------")
        except:
            print('---------------------------Can not load model----------------------------------')

    def store_transition(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
    
    def finish_path(self, next_state, done):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic_model(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()
    
    def discounted_cumulative_sums(self, x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def plot(max_step, rewards):
    if PLOT:
        # clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.title('episode. reward')
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Each Episode Reward')
        plt.savefig('reward_episode.png')
        plt.close('all')
        reward_c = pd.DataFrame(rewards)
        reward_c.to_csv("logs/origin.csv", index=None)
        # plt.show()
    else: return

    # current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # train_log_dir = "logs/"+current_time
    # summary_writer = tf.summary.create_file_writer(train_log_dir)
    # with summary_writer.as_default():
    #     tf.summary.scalar('Main/episode_reward', rewards, step=episodes)
    #     # tf.summary.scalar('Main/episode_steps', steps, step=episode)
    # summary_writer.flush()

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
            if RENDER :
                env.render()
            epoch_reward = 0
            step = 0
            Gt = 0
            train_count = 0

            if isinstance(state[ctrl_agent_index], type({})):
                obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
            else:
                obs_ctrl_agent, energy_ctrl_agent = state[ctrl_agent_index].flatten(), env.agent_list[ctrl_agent_index].energy
                obs_oppo_agent, energy_oppo_agent = state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

            # for t in range(MAX_STEPS):
            while True:
                step += 1

                action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
                # action_opponent = [0,0]  #here we assume the opponent is not moving in the demo

                action_ctrl= agent.get_action(obs_ctrl_agent) #action_ctrl[0] is force , action_ctrl[1] is angle
                action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

                # print("action is ", action_ctrl)
                next_state, reward, done, _ = env.step(action)
                # reward[0] += 0.1

                if isinstance(next_state[ctrl_agent_index], type({})):
                    next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index]['agent_obs'].flatten(), env.agent_list[ctrl_agent_index].energy
                    next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index]['agent_obs'], env.agent_list[1-ctrl_agent_index].energy
                else:
                    next_obs_ctrl_agent, next_energy_ctrl_agent = next_state[ctrl_agent_index], env.agent_list[ctrl_agent_index].energy
                    next_obs_oppo_agent, next_energy_oppo_agent = next_state[1-ctrl_agent_index], env.agent_list[1-ctrl_agent_index].energy

                if not done:
                    post_reward = [0, 0]
                else:
                    if reward[0] != reward[1]:
                        post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                    else:
                        post_reward=[0, 0]

                obs_oppo_agent, energy_oppo_agent = next_obs_oppo_agent, next_energy_oppo_agent
                obs_ctrl_agent, energy_ctrl_agent = np.array(next_obs_ctrl_agent).flatten(), next_energy_ctrl_agent

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

                agent.store_transition(obs_ctrl_agent, action_ctrl, reward[0])
                epoch_reward +=reward[0]

                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(next_obs_oppo_agent.flatten(), done)
                    agent.update()

                # try:
                #     if stop:
                #         time.sleep(2)
                #         raise ValueError ('stop from keyboard')
                # except Exception as e:
                #     print(str(e))
                #     raise ValueError ('stop from keyboard')

            agent.finish_path(next_obs_oppo_agent, done)
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