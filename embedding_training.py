import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

from src.utils import plot_confusion_matrix

def load_embedding_data(main_dir, subset='Train'):

    column_names = ['video_name', '??_1', '??_2', '??_3', 'segment_id', 'comb_label', 'audio_label', 'video_label',
                    'comb_det', 'audio_det', 'video_det', 'comb_pred', 'audio_pred', 'video_pred', 'embeddings']

    file_list = glob(os.path.join(main_dir, subset, '*'))

    df0 = pd.read_csv(file_list[0], sep="|", header=None)
    df1 = pd.read_csv(file_list[1], sep="|", header=None)
    df2 = pd.read_csv(file_list[2], sep="|", header=None)
    df3 = pd.read_csv(file_list[3], sep="|", header=None)

    df = pd.concat([df0, df1, df2, df3], ignore_index=True)
    df.columns = column_names

    # unwrap the embeddings
    X_pre = np.array(df['embeddings'])
    X_comb = np.zeros((len(X_pre), 320))
    for ii in range(len(X_pre)):
        X_comb[ii, :] = np.array([float(x) for x in X_pre[ii].split(',')])

    # divide the embeddings into numpy variables
    X_video = X_comb[:, :160]
    X_audio = X_comb[:, -160:]
    y_video = np.array(df['video_label'])
    y_audio = np.array(df['audio_label'])
    y_comb = np.array(df['comb_label'])

    # convert video label from source attrib to real/fake detection
    y_video = np.where(y_video == 0, 1, 0)

    return X_comb, X_audio, X_video, y_comb, y_audio, y_video


class monomodalNN(nn.Module):
    def __init__(self):
        super(monomodalNN, self).__init__()
        self.fc1 = nn.Linear(160, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class multimodalNN(nn.Module):
    def __init__(self):
        super(multimodalNN, self).__init__()
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


def evaluate_embedding_model(X_train, X_test, y_train, y_test, modal_save_path, modality='audio'):

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    # convert the numpy variables to tensors
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    # upload different models depending on the modality
    if modality in ['audio', 'video']:
        model = monomodalNN()
    else:
        model = multimodalNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    early_stopping_patience = 7
    early_stopping = 0
    best_val_loss = float('inf')

    model.train()
    for epoch in range(epochs):
        if early_stopping < early_stopping_patience:

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            y_train.view(-1).type(torch.int64)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()
                early_stopping = 0
            else:
                early_stopping += 1

    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, os.path.join(modal_save_path, f'{modality}_best_model.pth'))

    # EVALUATION

    model.eval()
    with torch.no_grad():
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        _,  outputs_pred = outputs.max(dim=1)
        balanced_accuracy = balanced_accuracy_score(y_test, outputs_pred)
        print(f"{modality.upper()} - Balanced Accuracy: {balanced_accuracy}")
        roc_auc = roc_auc_score(y_test, outputs[:, 1])
        print(f"{modality.upper()} - ROC AUC value: {roc_auc}\n")

        plt.figure(figsize=(6, 6))
        plot_confusion_matrix(y_test, outputs_pred, normalize=True)
        plt.show()

    print()



if __name__ == '__main__':

    main_dir = './data/embeddings_umur'
    modal_save_path = './checkpoints/embedding_model'

    X_train_comb, X_train_audio, X_train_video, y_train_comb, y_train_audio, y_train_video = load_embedding_data(main_dir, subset='Train')
    X_test_comb, X_test_audio, X_test_video, y_test_comb, y_test_audio, y_test_video = load_embedding_data(main_dir, subset='Test')

    evaluate_embedding_model(X_train_audio, X_test_audio, y_train_audio, y_test_audio, modal_save_path, modality='audio')
    evaluate_embedding_model(X_train_video, X_test_video, y_train_video, y_test_video, modal_save_path, modality='video')
    evaluate_embedding_model(X_train_comb, X_test_comb, y_train_comb, y_test_comb, modal_save_path, modality='multimodal')


    print()