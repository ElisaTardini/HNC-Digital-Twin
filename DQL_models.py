import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
from histories import *
from processing import *
from SVC_models import *

PATH = './Physician twin/'

def make_history_tensor(history, action):
    sample1 = history
    sample2 = history.multiply(action, axis='index')
    if len(history.shape) > 1:
        sample = np.concatenate((sample1.to_numpy(), sample2.to_numpy()), axis=1)
    else:
        sample = np.concatenate((sample1.to_numpy(), sample2.to_numpy()), axis=None)
    sample = torch.from_numpy(sample)
    return sample


def parse_history(history, action):
    if len(history.shape) > 1:
        sample = np.concatenate((history.to_numpy(), history.multiply(action, axis='index').to_numpy()), axis=1)

    else:
        sample = np.concatenate((history.to_numpy(), history.multiply(action, axis='index').to_numpy()),
                                axis=None).reshape(1, -1)

    return sample


class Q1_Net(nn.Module):

    def __init__(self):
        super(Q1_Net, self).__init__()
        self.fc1 = nn.Linear(48, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q2_Net(nn.Module):

    def __init__(self):
        super(Q2_Net, self).__init__()
        self.fc1 = nn.Linear(96, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q3_Net(nn.Module):

    def __init__(self):
        super(Q3_Net, self).__init__()
        self.fc1 = nn.Linear(132, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def compute_decision(history, action_values, net):
    q_max = None
    a_max = None
    for a in action_values:
        q = net(make_history_tensor(history, a)).item()

        if q_max is None:
            q_max = q
            a_max = a

        elif q > q_max:
            q_max = q
            a_max = a

    return a_max


def run_simulation(patient, histories, states, actions, action_values, models, trans_model, outcome_dicts, default_outcomes, derived_outcomes):
    as_doctor = True
    for i in range(len(actions)):
        action = compute_decision(patient[histories[i]], action_values[i], models[i])
        if (not as_doctor) or action != patient[actions[i]]:
            as_doctor = False
            patient = next_state(patient[histories[i]], actions[i], action, states[i], trans_model, outcome_dicts, default_outcomes, derived_outcomes)
    return patient


def treat(data, step):
    # Decision 1: (Induction Chemo) Y/N
    if step == 1:
        data[A1 + ' (Confidence %)'] = 0
        for i in range(1000):
            PATH1_i = f'{PATH}boot_{i}_OS_FT_AR_NoRad_deep_q1_2_layers_4.0_net.pth'
            q1_Net = Q1_Net()
            q1_Net.load_state_dict(torch.load(PATH1_i, map_location=torch.device('cpu')))
            q1_Net = q1_Net.double()
            for index, row in data.iterrows():
                data.loc[index, A1 + ' (Confidence %)'] += compute_decision(data.loc[index, H1], action_values[0], q1_Net)
        data[A1] = data[A1 + ' (Confidence %)'].apply(lambda x: 1 if x > 0 else -1)
        data[A1 + ' (Confidence %)'] = data[A1 + ' (Confidence %)'].apply(lambda x: (x + 1000) / 20 if x > 0 else (1000 - x) / 20)
    # Decision 2 (CC / RT alone)
    elif step == 2:
        data[A2 + ' (Confidence %)'] = 0
        for i in range(1000):
            PATH2_i = f'{PATH}boot_{i}_OS_FT_AR_NoRad_deep_q2_2_layers_4.0_net.pth'
            q2_Net = Q2_Net()
            q2_Net.load_state_dict(torch.load(PATH2_i, map_location=torch.device('cpu')))
            q2_Net = q2_Net.double()
            for index, row in data.iterrows():
                data.loc[index, A2 + ' (Confidence %)'] += compute_decision(data.loc[index, H2], action_values[1],
                                                                            q2_Net)
        data[A2] = data[A2 + ' (Confidence %)'].apply(lambda x: 1 if x > 0 else -1)
        data[A2 + ' (Confidence %)'] = data[A2 + ' (Confidence %)'].apply(
            lambda x: (x + 1000) / 20 if x > 0 else (1000 - x) / 20)
    # Decision 3 Neck Dissection (Y/N)
    elif step == 3:
        data[A3 + ' (Confidence %)'] = 0
        for i in range(1000):
            PATH3_i = f'{PATH}boot_{i}_OS_FT_AR_NoRad_deep_q3_2_layers_4.0_net.pth'
            q3_Net = Q3_Net()
            q3_Net.load_state_dict(torch.load(PATH3_i, map_location=torch.device('cpu')))
            q3_Net = q3_Net.double()
            for index, row in data.iterrows():
                data.loc[index, A3 + ' (Confidence %)'] += compute_decision(data.loc[index, H3], action_values[2],
                                                                            q3_Net)
        data[A3] = data[A3 + ' (Confidence %)'].apply(lambda x: 1 if x > 0 else -1)
        data[A3 + ' (Confidence %)'] = data[A3 + ' (Confidence %)'].apply(
            lambda x: (x + 1000) / 20 if x > 0 else (1000 - x) / 20)
    return data
