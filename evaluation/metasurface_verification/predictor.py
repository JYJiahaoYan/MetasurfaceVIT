
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import re
import json
from logger import create_logger


class Predictor:
    def __init__(self, args):
        self.wave = []
        self.num_wave = 0
        self.handled_wave_per_block = 1
        self.params_path = args.params_path
        self.designJM_path = args.designJM_path
        self.design_type = args.design_type
        self.treatment = args.treatment
        self.feature_string = args.feature_string
        self.lr = args.lr
        self.epoch = args.epoch
        self.get_params_from_design()
        self.hidden_channels = args.hidden_channels if args.hidden_channels else self.generate_hidden(args)
        # input size will always be 6 (3 for unitA, 3 for unitB)
        # for convnet1d, input channel is 1, the input will be [num_batch, 1, 6]
        self.network = args.network
        self.model = FullyConnectedNet(6, self.hidden_channels, self.num_wave * 6) if args.network == 'MLP' \
            else ConvNet1D(1, self.hidden_channels, self.num_wave, kernel_sizes=[3, 3, 3, 3, 3])
        self.is_train = args.train
        self.save_path = args.save_path
        self.save_name = ''
        self.data_path = args.path + args.finetune_folder
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = create_logger(output_dir=self.save_path, name="predictor_network")
        self.logger.info(self.__dict__)
        self.bipolar = args.bipolar
        self.batch_size = args.batch_size
        self.model_path = args.model_path  # load model for prediction & evaluation
        self.device = None

    def generate_hidden(self, args):
        """
        use if clause to choose if generate hidden_sizes for MLP or hidden_channels for CNN
        for MLP, the input size is 6, the output size is len(wave) * 6
        for CNN, the input channel is 1, the output channel is len(wave)
        """
        if args.network == 'MLP':
            rise_part = np.logspace(4, 9, 3, base=2, dtype=int)
            decline_part = np.logspace(8, np.log2(self.num_wave * 6 * 4), 3, base=2, dtype=int)
            return np.concatenate((rise_part, decline_part))
        else:
            rise_part = np.logspace(4, 8, 3, base=2, dtype=int)
            decline_part = np.logspace(7, np.log2(self.num_wave * 4), 2, base=2, dtype=int)
            return np.concatenate((rise_part, decline_part))

    # duplicate from Matcher
    def get_params_from_design(self):
        filename = self.designJM_path + '/type_' + str(self.design_type) + '_attr_' + self.treatment + '.json'
        try:
            with open(filename, 'r') as file:
                data = json.load(file)

            self.wave = data.get('wave')
            if isinstance(self.wave, int):
                self.num_wave = 1
            else:
                self.num_wave = len(self.wave)
            self.handled_wave_per_block = data.get('handled_wave_per_block')
            print(f"Attributes have been loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}.")

    def train(self):
        self.save_name = os.path.join(self.save_path, f"model_{self.network}_type_{self.design_type}_{self.treatment}.pth")
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloader, dataloader_val = self.load_train_val()
        self.train_runner(dataloader, dataloader_val)

    def select_lambda_from_whole(self, whole_JM, total_num):
        whole_JM = whole_JM.reshape((-1, total_num, 6))
        if self.num_wave == 1:
            return whole_JM[:, [self.wave], :]  # would be [num_sample, 1, 6] rather than [num_sample, 6]
        else:
            selected_JM_list = [whole_JM[:, i, :] for i in self.wave]
            selected_JM = np.stack(selected_JM_list, axis=1)
            self.logger.info(f'verify selected_JM dim: {np.shape(selected_JM)}')
            return selected_JM

    def load_train_val(self, val_ratio=0.8):
        X_path, Y_path, total_wave = '', '', ''
        for file in os.listdir(self.data_path):
            if file.startswith('param_double'):
                X_path = os.path.join(self.data_path, file)
            elif file.startswith('JM_double'):
                Y_path = os.path.join(self.data_path, file)
            # get the total num of wave info from finetune
            elif file.startswith('params_from'):
                content = read_to_list(os.path.join(self.data_path, file))
                for i in range(len(content)):
                    if content[i] == "DATA.SIZE_X":
                        total_wave = int(content[i + 1])
        if X_path == '' or Y_path == '' or total_wave == '':
            raise Exception("Can't find required txt data files. Please check the folder of finetune data.")
        X, whole_Y = np.loadtxt(X_path), np.loadtxt(Y_path)
        Y = self.select_lambda_from_whole(whole_Y, total_wave)
        if self.network == 'MLP':
            Y = Y.reshape((Y.shape[0], -1))
        if not self.bipolar:
            Y = 0.5 * Y + 0.5
        total_samples = len(X)
        train_samples = int(total_samples * val_ratio)  # 80% for training
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:]

        X_train = torch.from_numpy(X[train_indices]).float()
        Y_train = torch.from_numpy(Y[train_indices]).float()
        X_val = torch.from_numpy(X[val_indices]).float()
        Y_val = torch.from_numpy(Y[val_indices]).float()

        # Create the dataloader
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        dataset_val = torch.utils.data.TensorDataset(X_val, Y_val)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=True)

        return dataloader, dataloader_val

    def train_runner(self, dataloader, dataloader_val):
        self.model.to(self.device)
        loss_fn = F.mse_loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        for epoch in range(self.epoch):
            running_loss = 0.0
            running_loss_val = 0.0
            for i, data in enumerate(dataloader):
                # Forward pass
                inputs, labels = data
                # bug log: to(device) needs a return value
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Backward and optimize
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                # Update the running loss
                running_loss += loss.item()

                # Evaluate the model on the validation set
            with torch.no_grad():
                for i, data in enumerate(dataloader_val):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    val_loss = loss_fn(outputs, labels)
                    running_loss_val += val_loss.item()
            self.logger.info(
                f'Epoch [{epoch + 1}/{self.epoch}], Training Loss: {running_loss / len(dataloader):.4f}, '
                f'Validation Loss: {running_loss_val / len(dataloader_val):.4f}')
            # Save the model if the validation loss improves
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(self.model.state_dict(), self.save_name)
                self.logger.info(f'Model saved at epoch {epoch + 1}')

    def evaluate(self):
        dataloader = self.load_new()
        return self.evaluate_runner(dataloader)

    def load_new(self):
        paras_from_trans = np.loadtxt(self.params_path + '/type_' + str(self.design_type) + '_' + self.treatment + '.txt')
        paras_from_trans = torch.from_numpy(paras_from_trans).float()
        dataset = torch.utils.data.TensorDataset(paras_from_trans)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return dataloader

    # Load the best model and evaluate it
    def evaluate_runner(self, dataloader):
        self.model.to(self.device)
        if self.model_path == '':
            if not self.model_finder():
                raise ValueError("No state dict matches your model!")
        else:
            self.model.load_state_dict(torch.load(self.model_path))

        all_output = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs = data[0]
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # Append the current batch's outputs to the all_output list
                all_output.append(outputs.cpu().numpy())

        # Convert the list of outputs to a single numpy array
        all_output = np.concatenate(all_output, axis=0)
        return all_output

    def model_finder(self):
        skip = 0
        while True:
            self.model_path = self.find_latest_model(skip)
            if not self.model_path:
                print("No more models to try.")
                return False
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                print(f"Successfully loaded model from {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading model from {self.model_path}: {str(e)}")
                print("Trying the next latest model...")
                skip += 1

    def find_latest_model(self, skip=0):
        model_files = [f for f in os.listdir(self.save_path) if f.startswith('model_') and f.endswith('.pth')]
        if not model_files:
            return None

        sorted_models = sorted(model_files, key=lambda x: datetime.strptime(x[-14:-4], '%Y-%m-%d'), reverse=True)
        if skip < len(sorted_models):
            return os.path.join(self.save_path, sorted_models[skip])
        return None


class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FullyConnectedNet, self).__init__()

        # Define the layers of the network
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class ConvNet1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_output_channels, kernel_sizes, output_size=6):
        super(ConvNet1D, self).__init__()

        self.layers = nn.ModuleList()
        in_channels = input_channels

        for hidden_channel, kernel_size in zip(hidden_channels, kernel_sizes):
            self.layers.append(nn.Conv1d(in_channels, hidden_channel, kernel_size, padding=1))
            in_channels = hidden_channel

        self.final_conv = nn.Conv1d(in_channels, num_output_channels, 1)
        self.output_size = output_size

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.final_conv(x)
        x = F.adaptive_avg_pool1d(x, self.output_size)
        return x


def read_to_list(path):
    with open(path, 'r') as file:
        content = file.read()
        return content.split()
