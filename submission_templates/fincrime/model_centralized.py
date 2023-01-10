from pathlib import Path
#import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CentralizedModel:
    def __init__(self):
        self.xgb = XGBClassifier(n_estimators=100, max_depth = 7, base_score=0.01)
        self.lg =  LogisticRegression(random_state = 0)
        self.nn = Net_lg()
        #self.device = device

    def pre_process_swift(self, swift_data):
        # Hour
        swift_data["hour"] = swift_data["Timestamp"].dt.hour

        # Hour frequency for each sender
        senders = swift_data["Sender"].unique()
        swift_data["sender_hour"] = swift_data["Sender"] + swift_data["hour"].astype(str)
        sender_hour_frequency = {}
        for s in senders:
            sender_rows = swift_data[swift_data["Sender"] == s]
            for h in range(24):
                sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])

        swift_data["sender_hour_freq"] = swift_data["sender_hour"].map(sender_hour_frequency)

        # Sender-Currency Frequency and Average Amount per Sender-Currency
        swift_data["sender_currency"] = swift_data["Sender"] + swift_data["InstructedCurrency"]

        sender_currency_freq = {}
        sender_currency_avg = {}

        for sc in set(
            list(swift_data["sender_currency"].unique()) #+ list(swift_test["sender_currency"].unique())
        ):
            sender_currency_freq[sc] = len(swift_data[swift_data["sender_currency"] == sc])
            sender_currency_avg[sc] = swift_data[swift_data["sender_currency"] == sc][
                "InstructedAmount"
            ].mean()

        swift_data["sender_currency_freq"] = swift_data["sender_currency"].map(sender_currency_freq)

        swift_data["sender_currency_amount_average"] = swift_data["sender_currency"].map(
            sender_currency_avg
        )

        # Sender-Receiver Frequency
        swift_data["sender_receiver"] = swift_data["Sender"] + swift_data["Receiver"]

        sender_receiver_freq = {}

        for sr in set(
            list(swift_data["sender_receiver"].unique()) #+ list(swift_test["sender_receiver"].unique())
        ):
            sender_receiver_freq[sr] = len(swift_data[swift_data["sender_receiver"] == sr])

        swift_data["sender_receiver_freq"] = swift_data["sender_receiver"].map(sender_receiver_freq)

        return swift_data
    
    def combine_swift_and_bank(self, swift_data, bank_data):
        # combine the table and add flag features columns
        combine = (
        swift_data.reset_index().rename(columns={'index': 'MessageId'})
            .merge(
                right=bank_data[["Bank", "Account", "Flags"]].rename(
                    columns={"Flags": "OrderingFlags"}
                ),
                how="left",
                left_on=["OrderingAccount"],
                right_on=["Account"],
            )
            .set_index("MessageId")
        )
        combine = (
        combine.reset_index().rename(columns={'index': 'MessageId'})
            .merge(
                right=bank_data[["Bank", "Account", "Flags"]].rename(
                    columns={"Flags": "BeneficiaryFlags"}
                ),
                how="left",
                left_on=["BeneficiaryAccount"],
                right_on=["Account"],
            )
            .set_index("MessageId")
        )

        # drop the columns thats not useful in training XGBoost
        columns_to_drop = [
            "UETR",
            "Sender",
            "Receiver",
            "TransactionReference",
            "OrderingAccount",
            "OrderingName",
            "OrderingStreet",
            "OrderingCountryCityZip",
            "BeneficiaryAccount",
            "BeneficiaryName",
            "BeneficiaryStreet",
            "BeneficiaryCountryCityZip",
            "SettlementDate",
            "SettlementCurrency",
            "InstructedCurrency",
            "Timestamp",
            "sender_hour",
            "sender_currency",
            "sender_receiver",
            "Bank_x",
            "Bank_y",
            "Account_y",
            "Account_x",
        ]

        combine = combine.drop(columns_to_drop, axis=1)
        return combine

    def transform_and_normalized(self, combine):
        Y = combine["Label"].values
        X = combine.drop(["Label"], axis=1).values

        # Normalize
        scaler = StandardScaler()
        scaler.fit(X)

        X = scaler.transform(X)
        return X, Y
    
    def get_X_swift(self, X):
        X_swift = []
        for i in range(len(X)):
            X_swift.append(X[i][:-2])
        X_swift = np.asarray(X_swift)
        return X_swift
    
    def get_X_logistic_regression(self,X, pred_proba_xgb):
        X_lg = []
        for idx in range(len(X)):
            temp = X[idx][-2:]
            temp = np.append(temp,pred_proba_xgb[idx])
            X_lg.append(temp)
        X_lg = np.asarray(X_lg)
        X_lg = np.nan_to_num(X_lg, nan=12)
        return X_lg
    
    def get_dataloader_for_NN(self,X_lg, Y):
        set = TrainData(torch.FloatTensor(X_lg), torch.FloatTensor(Y))
        dataloader = DataLoader(set, batch_size=32)
        return dataloader
    
    def train_NN(self, train_loader, device):
        self.nn.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=0.1)
        avg_acc = 0
        avg_loss = 0
        total = 0

        self.nn.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(1)

            optimizer.zero_grad()
            output = self.nn(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # get statistics
            predicted = torch.round(output)
            correct = (predicted == target).sum()
            avg_acc += correct.item()
            avg_loss += loss.item() * data.shape[0]
            total += data.shape[0]

        return avg_acc/total, avg_loss/total
    
    def test_NN(self, test_loader, device):
        self.nn.to(device)
        criterion = nn.BCELoss()
        y_proba_list = []
        self.nn.eval()
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                target = target.unsqueeze(1)
                output = self.nn(data)
                loss = criterion(output, target)

                # get statistics
                y_proba_list.extend(output.detach().cpu().numpy())
        return y_proba_list

    def save(self, path):
        xgb_path = os.path.join(path, "centralized_xgboost.pkl")
        #lg_path = os.path.join(path,"centralized_lg.pkl")
        nn_path = os.path.join(path,"centralized_nn.pkl")
        #joblib.dump(self.xgb, xgb_path)
        #joblib.dump(self.lg, lg_path)
        with open(xgb_path, 'wb') as f:
            pickle.dump(self.xgb, f)
        #with open(lg_path, 'wb') as f:
        #    pickle.dump(self.lg, f)  
        with open(nn_path, 'wb') as f:
            pickle.dump(self.nn, f) 
    

    @classmethod
    def load(cls, path):
        inst = cls()
        #inst.pipeline = joblib.load(path)
        xgb_path = os.path.join(path,"centralized_xgboost.pkl")
        #lg_path = os.path.join(path,"centralized_lg.pkl")
        nn_path = os.path.join(path,"centralized_nn.pkl")

        with open(xgb_path, 'rb') as f:
            inst.xgb = pickle.load(f)
        #with open(lg_path, 'rb') as f:
        #    inst.lg = pickle.load(f)
        with open(nn_path, 'rb') as f:
            inst.lg = pickle.load(f)
        return inst


class Net_lg(nn.Module):
    def __init__(self):
        super(Net_lg, self).__init__()
        self.layer_1 = nn.Linear(3, 1) 
        self.sigmoid =  nn.Sigmoid()
        
        
    def forward(self, inputs):
        x = self.sigmoid(self.layer_1(inputs))
        return x
    

class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)  
