import random

import numpy as np
import glob

import torch
import utils
import time
import glob
import config
class Dataset():
    # fro this dataset, the subgoal_trajectory count is 1.
    def __init__(self, cur_pth, args):
        self.args = args
        self.expert_data = np.load(cur_pth+args.expert.basic_tra, allow_pickle=True).item()
        self.states = self.expert_data["states"]
        self.action = self.expert_data["actions"]
        self.next_states = self.expert_data["next_states"]
        self.all_data = np.load(cur_pth+args.expert.tra,allow_pickle=True).item()
        self.all_state = np.vstack([self.all_data["states"],self.states]).reshape(-1,self.states.shape[2])
        self.all_action = np.vstack([self.all_data["actions"],self.action]).reshape(-1,self.action.shape[2]).clip(-1.0,1.0)
        # extract self state
        self.is_done = self.expert_data["dones"]
        self.tra_num = self.expert_data["dones"].shape[0]
        self.goals = np.zeros((self.tra_num),dtype=np.int32) ## goals: state id
        for i in range(self.tra_num):  ## TODO: Set random goal at start
            # self.goals[i] = random.randint(1,args.first_step)
            self.goals[i] = random.randint(1,args.env.first_step)
        self.belongs = np.zeros(self.expert_data["rewards"].shape, dtype=np.int16) ##
        self.reset_belongs()
        self.insert_new_subgoal()

    def reset_belongs(self):
        for i in range(self.tra_num):
            for j in range(self.expert_data["lengths"][i]):
                # if j != 0:
                #     self.belongs[i][j] = min( random.randint(max(j + 1, self.belongs[ i][j-1]), j+self.args.env.first_step), self.states[0].shape[0]-1)
                # else:
                #     self.belongs[i][j] = min( random.randint(j + 1, j + self.args.env.first_step), self.states[0].shape[0] - 1)
                self.belongs[i][j] = min(j+1,self.states[0].shape[0]-1)
    def insert_new_subgoal(self,pos=None):# pos list
        # self.goals: tra_num
        for i in range(self.tra_num):
            if pos != None:
                if pos[i]< self.goals[i]:
                    print("??? There is some error in subgoal setting at trajectory %d, goal: %d , new goal: %d" %(i, self.goals[i], pos[i] ) )
                    return False
                self.goals[i]=pos[i]
            for j in range(self.goals[i]):
                self.belongs[i][j] = self.goals[i]
            print(self.goals[i])
            for j in range(self.expert_data["lengths"][i]):
                print(self.belongs[i][j], end=' ')
        return True
    def find_subgoal(self, state):
        # state: batch_num * state_dim
        # return: batch_num * state_dim
        focus_state=state[:,self.args.env.l_ego_s:self.args.env.r_ego_s+1]
        subgoals=np.zeros(state.shape)
        for i in range(state.shape[0]):
            id=0
            min_dis = float('inf')
            for j in range(self.states[0].shape[0]): #TODO:there is only one trajectory to get subgoal!
                dis=np.linalg.norm(focus_state[i]-self.states[0][j][self.args.env.l_ego_s:self.args.env.r_ego_s+1])
                if dis <= min_dis:
                    min_dis = dis
                    id = self.belongs[0][j]
            subgoals[i]=self.states[0][id]
        return id, subgoals

    def sample(self,device):
        batch_size = 32
        indexes = np.random.choice(np.arange(self.expert_data["lengths"][0]), size=batch_size, replace=False)
        batch_state, batch_action = [self.states[0][i] for i in indexes], [self.action[0][i] for i in indexes]
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        return batch_state, batch_action

    def all_sample(self,device):
        batch_size = 32
        indexes = np.random.choice(np.arange(self.all_state.shape[0]), size=batch_size, replace=False)
        batch_state, batch_action = [self.all_state[i] for i in indexes], [self.all_action[i] for i in indexes]
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        return batch_state, batch_action
    def sqil_sample(self,device):
        batch_size = 32
        indexes = np.random.choice(np.arange(self.all_state.shape[0]), size=batch_size, replace=False)
        batch_state, batch_action ,batch_next_state, batch_done = [self.all_state[i] for i in indexes], [self.all_action[i] for i in indexes], [self.all_state[min(i+1,self.expert_data["lengths"][0]-1)] for i in indexes], [1.0 if i==self.expert_data["lengths"][0]-1 else 0.0 for i in indexes]
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_action)
        batch_next_state = np.array(batch_next_state)
        batch_done = np.array(batch_done)
        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device)
        return batch_state, batch_action, batch_next_state,batch_done
    def get_tra_num(self):
        return self.all_data["dones"].shape[0]

    def select_subgoal(self, agent, args):
        flag = False
        test_s = torch.squeeze(torch.tensor(self.states).float(), dim=0)
        if args.insert_subgoal_exp == True:
            test_a = torch.squeeze(torch.tensor(self.action).float(), dim=0)
        else:
            test_a = agent.choose_action(test_s, sample=True)
            test_a = torch.squeeze(torch.tensor(test_a).float(), dim=0)
        UC = agent.getUC(test_s, test_a).squeeze()
        target_uc = args.env.delta
        base_uc = UC[self.goals[0]]
        cnt = 1.0
        for i in range(self.goals[0] + 1, self.expert_data["lengths"][0]):
            if (args.soft_mean and base_uc * target_uc < UC[i]) or (
                    (not args.soft_mean) and base_uc * target_uc / cnt < UC[i]):  # TODO more specific condition?:
                pos = np.array(i, dtype=np.int16).reshape((1, 1))
                # pos += random.randint(1,args.env.next_step)
                flag = self.insert_new_subgoal(pos)
                break
            base_uc = base_uc * args.sigma + (1 - args.sigma) * UC[i] if args.soft_mean else base_uc + UC[i]
            cnt += 1.0 if args.soft_mean else 0.0
        print(self.goals[0])
        return flag
def get_reward(state):
    return None
