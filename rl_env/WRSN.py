import yaml
import copy
import gym
from gym import spaces
import torch

import numpy as np
import sys
import os
from scipy.spatial.distance import euclidean
root_dir = os.getcwd()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from rl_env.state_representation.GNN import GCN
from rl_env.state_representation.StateRepresentation import GraphRepresentation
    
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent, warm_up_time = 100):
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.warm_up_time = warm_up_time
        self.epsilon = 1e-9
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64)
        self.agents_process = [None for _ in range(num_agent)]
        self.agents_action = [None for _ in range(num_agent)]
        self.agents_prev_state = [None for _ in range(num_agent)]
        self.agents_exclusive_reward = [0 for _ in range(num_agent)]
        self.reset()
        # create graph
        self.graph = GraphRepresentation.get_graph_representation(self.net)

    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate()) & self.env.process(self.update_reward())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
            agent.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0]
        self.moving_time_max = (euclidean(np.array([self.net.frame[0], self.net.frame[2]]), np.array([self.net.frame[1], self.net.frame[3]]))) / self.agent_phy_para["velocity"]
        self.charging_time_max = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) / (self.agent_phy_para["alpha"] / (self.agent_phy_para["beta"] ** 2))
        self.avg_nodes_agent = (self.net.nodes_density * np.pi * (self.agent_phy_para["charging_range"] ** 2))
        self.env.run(until=self.warm_up_time)
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True
        for id, agent in enumerate(self.agents):
            self.agents_prev_state[id] = self.get_state(agent.id)
            self.agents_action[id] = np.array([0.5, 0.5, 0])
            self.agents_process[id] = self.env.process(self.agents[id].operate_step(copy.deepcopy(agent.cur_phy_action)))
            self.agents_exclusive_reward[id] = 0.0  

        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id":id, 
                        "prev_state": self.agents_prev_state[id],
                        "action":self.agents_action[id], 
                        "reward": 0.0,
                        "state": self.agents_prev_state[id],
                        "terminal":tmp_terminal,
                        "info": [self.net, self.agents]}
        return {"agent_id":None, 
                "prev_state": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal":tmp_terminal,
                "info": [self.net, self.agents]}
    
    def config_action(self, agent_id, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                self.charging_time_max * action[2]])
    
    def update_reward(self):
        """_summary_
        Hàm update_reward nhằm:
        - Xác định mức độ ưu tiên cho việc sạc của các node trong mạng.
        - Tính toán điểm thưởng cho mỗi tác nhân dựa trên sự cải thiện năng lượng của các node mà nó ảnh hưởng.
        - Cập nhật điểm thưởng độc quyền (agents_exclusive_reward) của từng tác nhân, giúp đánh giá hiệu suất của chúng trong việc kéo dài thời gian sống của các node.
        """
        yield self.env.timeout(0)
        
    def get_state(self, agent_id):
        """
        :param agent_id:
        :return:
        """
        model_path = os.path.join(root_dir, "rl_env", "grap_model.pth")
        data = GraphRepresentation.create_graph(self.net)
        num_features = data.x.size(1)
        num_classes = len(self.net.listChargingLocations) + 1
        hidden_dim = 512
        output_dim = 83
        GNN_model = GCN(num_features, hidden_dim, output_dim, num_classes)

        GNN_model.load_state_dict(torch.load(model_path))
        GNN_model.eval()
        with torch.no_grad():
            _, embeddings = GNN_model(data.x, data.edge_index)
        enegy = self.get_enegy()
        embeddings = torch.cat((embeddings, enegy), 1)
        return embeddings
    def get_reward(self, agent_id = 0):
        """_summary_
        Đánh giá hiệu quả của một tác nhân trong việc:
        - Cải thiện hoặc duy trì thời gian của mạng (toàn bộ node).
        - Đóng góp riêng cho các node mà tác nhân đó trực tiếp sạc.
        Args:
            agent_id (_type_): id của tác tử được tính toán phần thưởng

        Returns:
            _type_: double
            Phần thưởng của tác nhân
        """




        return None
    
    def step(self, agent_id, input_action):
        if agent_id is not None:
            action = np.array(input_action)
            self.agents_action[agent_id] = action
            self.agents_process[agent_id] = self.env.process(self.agents[agent_id].operate_step(self.config_action(agent_id, action)))
            self.agents_prev_state[agent_id] = self.get_state(agent_id)
            self.agents_exclusive_reward[agent_id] = 0

        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]
        self.env.run(until=general_process)
        print(self.env.now)
        if self.net.alive == 0:
            return {"agent_id":None, 
                    "prev_state": None,
                    "action":None, 
                    "reward": None,
                    "state": None,
                    "terminal":True,
                    "info": [self.net, self.agents]}
        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < self.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id": id, 
                        "prev_state": self.agents_prev_state[id],
                        "action":self.agents_action[id], 
                        "reward": self.get_reward(id),
                        "state": self.get_state(id), 
                        "terminal": False,
                        "info": [self.net, self.agents]}