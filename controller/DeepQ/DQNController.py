
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from DQN_model import DQN
from ReplayBuffer import ReplayBuffer

class DQNController:
    def make_action(self, id, state, info, wrsn):
        pass
    def __init__(self):
        pass
    def train(self):
        pass