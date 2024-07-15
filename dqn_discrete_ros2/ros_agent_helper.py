from state_subscriber import StateSubscriber
from random_action_subscriber import RandomActionSubscriber
from action_publisher import ActionPublisher
import random

class RosAgentHelper:
    def __init__(self):
        # declare new action publisher
        # declare random action subscriber
        # declare state_subscriber
        self.state_subscriber = StateSubscriber()
        self.random_action_subscriber = RandomActionSubscriber()
        self.action_publisher = ActionPublisher() 

    def step(self):
        self.current_state = self.state_subscriber.step_data()
        
        # questions for below
        # how do we determine if we are training or not
        # random values are taken into account in the model, how will that work/be updated?
        # epsilon value and epsilon decay: both values from yaml file. how do we implement the decay
        # decay implemnetation is not major priority

        # randomly pick random action, or use action from model
        if is_training and random.random() < epsilon: # if random value is less then an epsilon value, choose random action
            action = self.random_action_subscriber.step_data()
        else:
            action = self.get_dqn_action(self.current_state)
        
        # Publish the selected action
        self.action_publisher.publish_action(action)
    
    # in a case where a random action is not picked. we want to 
    def get_dqn_action(self, state):
        #dqn model stuff
        return selected_action