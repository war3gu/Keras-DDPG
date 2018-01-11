#https://gym.openai.com/evaluations/eval_f1UA1ayyQnKlGtPqk8IJQ
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
import gym
from gym import wrappers
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

def playGame(train_indicator = 1):
    BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    GAMMA = 0.9
    TAU = 0.01
    lr_actor = 1e-3
    lr_critic = 1e-3
    train_interval = 1
    train_times = 20
    action_dim = 3
    state_dim = 2
    
    np.random.seed(2333)
    
    EXPLORE = 5000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    LOSS = 0
    
    
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, lr_actor)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, lr_critic)
    buff = ReplayBuffer(BUFFER_SIZE)
    
    env = gym.make('MountainCarContinuous-v0')
    # env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1')
    
    # Now load the weight
    print("Now we load the weight")
    try:
        # actor.model.load_weights("actormodel.h5")
        # critic.model.load_weights("criticmodel.h5")
        # actor.target_model.load_weights("actormodel.h5")
        # critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    loss = 0
    for i in range(episode_count):
        # print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        ob = env.reset()
        s_t = ob
        total_reward = 0.
        for j in range(max_steps):
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            for k in range(action_dim):
                noise_t[0][k] = train_indicator * max(epsilon, 0) * OU().function(a_t_original[0][k], 0, 1.0, 0.3)
            action = a_t_original[0]
            env.render()
            ob, r_t, done, _ = env.step(action)
            s_t1 = ob
            # print(ob)
            buff.add(s_t, a_t_original[0], r_t, s_t1, done)
            
            total_reward += r_t
            s_t = s_t1
            step += 1
            if done:
                print("Episode", i, "Step", step, "Reward", total_reward)
                break
        if (train_indicator) and i % train_interval == 0:
            loss = 0
            for T in range(train_times):
                batch = buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.asarray([0.0 for e in batch])
                    
                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
                    
                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA * target_q_values[k]
                loss = critic.model.train_on_batch([states,actions], y_t)
                
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()                
            print("Episode", i, "Step", step, "Loss", loss)
        if np.mod(i, 3) == 0:
            if (train_indicator):
            # print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)
        
                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

#print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward), "loss: " + str(LOSS), "epsilon" + str(epsilon))
    #       print("Total Step: " + str(step))
    #       print("")
    
    print("Finish.")
    env.close()
#    gym.upload('/tmp/cartpole-experiment-1', api_key='sk_PbPX46F2SqmECJp08RykGQ')

if __name__ == "__main__":
    playGame()
