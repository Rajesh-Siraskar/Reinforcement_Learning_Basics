#-----------------------------------------------------------------------------------------------------------------
# Milling Tool wear environment 
# Author: Rajesh Siraskar
# Date: 28-Apr-2023
#-----------------------------------------------------------------------------------------------------------------

import gym
from gym import spaces
import pandas as pd
import numpy as np


class MillingTool(gym.Env):
    """Custom Milling Tool Wear Environment that follows the Open AI gym interface."""

    metadata = {"render.modes": ["human"]}
    
    def __init__(self, df, wear_threshold, wear_max, replacement_threshold):        

        # Machine data frame properties    
        self.df = df
        self.df_length = len(self.df.index)
        self.df_index = 0
        
        self.wear_threshold = wear_threshold
        self.wear_max = wear_max
        self.replacement_threshold = replacement_threshold
        self.reward = 0.0
        
        self.ep_total_reward = 0
        self.ep_length = 0
        self.ep_tool_replaced = 0
        self.ep_avg_reward = []
        self.ep_avg_length = []
        self.ep_tool_replaced_history = []
        
        # All features are normalized [0, 1]
        self.min_feature = 0.0
        self.max_feature = 1.0
        
        # Define state and action limits
        # For more features use: For eg. 3 features
        # low_state = np.array([min_feature, min_feature, min_feature], dtype=np.float32)
        # similarly for high_state - just add comma separated max values, for as many features used
        
        self.low_state = np.array([self.min_feature, self.min_feature], dtype=np.float32)
        self.high_state = np.array([self.max_feature, self.max_feature], dtype=np.float32)
        
        # Observation and action spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
             
    def step(self, action):
        """
        Args: action. Continous value, which is the predicted RUL in our case
        Returns: next_state, reward, terminated, truncated , info
        """
        terminated = False
        info = {'action':'Continue'} 
        self.state = self._get_observation(self.df_index)
        time_step = self.state[0]
        wear = self.state[1]
        
        # Reward function:
        #       Below threshold: + [e^(VB-Threshold)]/2
        #       Above threshold: - e^(VB-Threshold)
        #       Clipped to: -100.0 to +100.0
        
        if wear < self.wear_threshold:
            reward = 1.0
            #reward = np.exp(wear - self.wear_threshold)/2
        else:
            reward = -10.0
            #reward = -np.exp(wear - self.wear_threshold)
        
        #self.reward += float(np.clip(reward, -100.00, 10.00))
        
        # Based on the action = 1 replace the tool or if 0, continue with normal operation
        if action:
            # print(f'-- Tool replaced at {self.df_index:>3d} | Reward: {self.reward:>7.2f} | Wear: {wear:>5.4f} | Action: {info}')
            # We replace the tool - so roll back tool life. -1 so that the increment in df_index will reset it to 0
            self.df_index = -1
            self.ep_tool_replaced += 1
            reward = -2.0
            info = {'action':'Tool replaced'} 

        # Even if action was to replace, if we had crossed the absolute maximum, then we need to terminate this episode
        if wear > self.wear_max*self.wear_threshold:
            terminated = True
            reward = 0.0
            
        self.reward += float(reward)
        
        # Post process of step: Get next observation, fill history arrays
        self.df_index = self.df_index + 1
        if self.df_index > (self.df_length-2):
            terminated = True
            
        # We can now read the next state, for agent's policy to predict the "Action"
        state_ = self._get_observation(self.df_index)
            
        self.ep_total_reward += reward
        self.ep_length += 1
        return state_, self.reward, terminated, info

    def _get_observation(self, index):
        next_state = np.array([
            self.df['time'][index],
            self.df['VB_mm'][index]
        ], dtype=np.float32)
                
        return next_state

    def reset(self):
        #print('\n', 80*'-')
        if self.ep_length > 0:
            self.ep_avg_reward.append(self.ep_total_reward/self.ep_length)
            self.ep_avg_length.append(self.ep_length)
            self.ep_tool_replaced_history.append(self.ep_tool_replaced)
            
        self.ep_total_reward = 0
        self.ep_tool_replaced = 0
        self.ep_length = 0
    
        terminated = False
        self.df_index = 0
        self.reward = 0.0
        self.state = self._get_observation(self.df_index)
         
        return np.array(self.state, dtype=np.float32)
    
    def render(self, mode='human', close=False):
        print(f'-- Index: {self.df_index:4d}: Reward: {self.reward:>12.3f}')
        
    def close(self):
        del [self.df]
        gc.collect()
        print('** -- Envoronment closed. Data-frame memory released. Garbage collector invoked successfully -- **')
        
        