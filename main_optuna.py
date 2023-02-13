#! /usr/bin/env python3

'''
significant DQN/Pytorch code copied from https://github.com/RoyElkabetz/DQN_with_PyTorch_and_Gym
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import rospy
import argparse, os
import copy as cp
from utils import plot_learning_curve
import agents as Agents
import time
import drone_gym_gazebo_env
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Drone StableBaselines')
    parser.add_argument('-train', type=int, default=1, help='True = training, False = playing')
    parser.add_argument('-load_checkpoint', type=int, default=0, help='Load a model checkpoint')
    #parser.add_argument('-save', type=bool, default=False, help='If want to save files make true')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation.')    
    parser.add_argument('-epsilon', type=float, default=1, help='What epsilon starts at')    
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-max_mem', type=int, default=20000, help='Maximum size for memory replay buffer')    
    parser.add_argument('-bs', type=int, default=128, help='Batch size for replay memory sampling')    
    parser.add_argument('-eps_min', type=float, default=0.1, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-eps_dec', type=float, default=2*1e-4, help='Linear factor for decreasing epsilon')#1e-3=1500steps,1e-4=0.0001,1e-5, 1e-4=0.0001,600 games
    parser.add_argument('-replace', type=int, default=1000, help='Interval for replacing target network')
    parser.add_argument('-algo', type=str, default='DuelingDDQNAgent', choices=['DQNAgent', 'DDQNAgent', 'DuelingDQNAgent', 'DuelingDDQNAgent'])
    parser.add_argument('-root', type=str, default='/home/patmc/catkin_ws/src/mavros-px4-vehicle/', help='root path for saving/loading')    
    parser.add_argument('-path', type=str, default='/home/patmc/catkin_ws/src/mavros-px4-vehicle/models/', help='path for model saving/loading')    
    parser.add_argument('-n_games', type=int, default=250)#200=5000steps
    parser.add_argument('-total_steps', type=int, default=500)#20000)    
    parser.add_argument('-render', type=bool, default=False)
    args = parser.parse_args()

    rospy.init_node('train_node', anonymous=True)

    env_name = 'DroneGymGazeboEnv-v0'
    env = gym.make(env_name)

    trial_number = 0

    def objective(trial):
        global env_name, env, trial_number
        rospy.logwarn("############################################################")
        rospy.logwarn("############################################################")
        rospy.logwarn("Trial number = " + str(trial_number))
        rospy.logwarn("############################################################")
        rospy.logwarn("############################################################")

        best_score = -np.inf
        agent_ = getattr(Agents, args.algo)
        agent = agent_(gamma=args.gamma,#trial.suggest_categorical("gamma", [0.9, 0.99]),
                       epsilon=args.epsilon,
                       lr=trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3]),#suggest_float("lr", 0.00001, 0.001),#args.lr,
                       n_actions=env.action_space.n,
                       image_input_dims=(env.image_observation_space.shape),
                       mem_size=args.max_mem,
                       batch_size=trial.suggest_categorical("batch_size", [32, 128]),#args.bs,
                       eps_min=args.eps_min,
                       eps_dec=args.eps_dec,#1e-4,600 games
                       replace=args.replace,#trial.suggest_categorical("replace", [100, 1000]),#args.replace,
                       total_steps=args.total_steps,
                       algo=args.algo,
                       env_name=env_name,
                       chkpt_dir=args.path)

        # intializations:
        n_steps = 0
        scores, eps_history, steps_array, times_array = [], [], [], []
        #total_steps_to_take = args.total_steps
        #games_played = 0
        #reached_total_steps = False

        # setup files for saving/loading:
        #fname = args.algo + '_1_lr' + str(args.lr) + '_bs' + str(args.bs) + '_gamma' + str(args.gamma) + '_episodes' + str(args.n_games)
        fname = args.algo + '_env1A' + '_trial' + str(trial_number) + '_lr' + str(agent.lr) + '_bs' + str(agent.batch_size) + '_episodes' + str(args.n_games)
        log_path = args.root + 'logs/'
        scores_file = args.root + 'scores/' + fname + '_scores.npy'
        steps_file = args.root + 'scores/' + fname + '_steps.npy'
        eps_history_file = args.root + 'scores/' + fname + '_eps_history.npy'
        figure_file_start = args.root + 'plots/' + fname
        figure_file = figure_file_start + '.png'    
        times_file = args.root + 'times/' + fname + '_times.txt'

        if args.train:
            tensorboard_writer = SummaryWriter(log_dir=log_path)

        if args.load_checkpoint:
            agent.load_models() #load Q models

        # training / playing
        start_time = time.time()
        #episode = 0
        #while not reached_total_steps:
        for episode in range(args.n_games):#games_played, args.n_games + games_played):
            episode += 1 #want first episode to be one not zero
            done = False
            score = 0
            observation = env.reset()
            rospy.logwarn("##############################")
            rospy.logwarn("Trial number = " + str(trial_number) + ", Episode = " + str(episode))
            rospy.logwarn("##############################")
            while not done:
                rospy.loginfo("Episode = " + str(episode) + ", n_steps = " + str(n_steps))
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                
                #if n_steps >= total_steps_to_take:
                #    done = True
                #    reached_total_steps = True

                score += reward
                if args.render:
                    env.render()
                if args.train:
                    agent.store_transition(observation, action, reward, observation_, int(done))
                    agent.learn()
                observation = observation_
                n_steps += 1

            scores.append(score)#scores equivalent to total_rewards and score equiv to reward
            steps_array.append(n_steps)
            eps_history.append(agent.epsilon)
            mean_score = np.mean(scores[-100:])

            if args.train:
                tensorboard_writer.add_scalar('Reward vs Timesteps', score, n_steps)
                tensorboard_writer.add_scalar('Reward vs Episodes', score, episode)
                tensorboard_writer.add_scalar('Average Reward vs Timesteps', mean_score, n_steps)
                tensorboard_writer.add_scalar('Average Reward vs Episodes', mean_score, episode)           
                tensorboard_writer.add_scalar('Epsilon vs Timesteps', agent.epsilon, n_steps)
                tensorboard_writer.add_scalar('Epsilon vs Episodes', agent.epsilon, episode)

            print('episode ', episode, 'score: ', score, 'average score %.1f best average score %.1f epsilon %.2f' %
                  (mean_score, best_score, agent.epsilon), 'steps ', n_steps)

            if mean_score > best_score:
                if args.train:
                    agent.save_models()
                best_score = mean_score

        if args.train:
            # save training data
            times_array.append(round(time.time() - start_time, 2))
            #np.save(scores_file, np.array(scores))
            #np.save(steps_file, np.array(steps_array))
            #np.save(eps_history_file, np.array(eps_history))
            np.savetxt(times_file, times_array, delimiter=',')  
            # plot the learning curve:
            plot_learning_curve(steps_array, scores, eps_history, figure_file)
            #close tensorboard_writer:
            tensorboard_writer.close()

        trial_number += 1
        return np.mean(scores)

    study = optuna.create_study(study_name="D3QN Drone Gym Gazebo", direction="maximize")#, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=6)#, catch=(RuntimeError,),  gc_after_trial=True)

    # Make the drone land.
    env.land_disconnect_drone()
    rospy.logwarn("end")