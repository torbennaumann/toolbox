import json
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class TrainClassifier():
    def __init__(self, model, inputs, targets, exp_dir):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.exp_dir = exp_dir

        self.data_is_prepared = False


        # define the name of the directory to be created
        if(not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)


    def prepare_data(self, test_size=0.2):
        inputs_train, inputs_val, targets_train, targets_val = train_test_split(self.inputs, self.targets, test_size=test_size)

        self.N = inputs_train.shape[0]

        self.x = torch.tensor(inputs_train.values, device=self.model.device, dtype=self.model.dtype)
        self.y = torch.tensor(targets_train.values, device=self.model.device, dtype=torch.long).squeeze()

        self.x_val = torch.tensor(inputs_val.values, device=self.model.device, dtype=self.model.dtype)
        self.y_val = torch.tensor(targets_val.values, device=self.model.device, dtype=torch.long).squeeze()

        del inputs_train
        del inputs_val
        del targets_train
        del targets_val

        self.data_is_prepared = True
        return


    def run_train(self, n_epochs, lr=0.001, batch_size=256):
        if(self.data_is_prepared == False):
            self.prepare_data()

        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Train
        loss_hist = []
        loss_validate_hist = []

        for t in range(n_epochs):
            for batch in range(0, int(self.N / batch_size)):
                # Berechne den Batch
                batch_x = self.x[batch * batch_size: (batch + 1) * batch_size, :]
                batch_y = self.y[batch * batch_size: (batch + 1) * batch_size]

                # Berechne die Vorhersage (foward step)
                outputs = self.model(batch_x)

                # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
                loss = criterion(outputs, batch_y)

                # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
            if t % 10 == 0:
                outputs = self.model(self.x)
                loss = criterion(outputs, self.y)
                loss_hist.append(loss.item())

                outputs_val = self.model(self.x_val)
                loss_val = criterion(outputs_val, self.y_val)
                loss_validate_hist.append(loss_val.item())

                print(t, ' train_loss: ',loss.item(), 'validate_loss: ', loss_val.item())

        torch.save(self.model, self.exp_dir + self.model.name + '.pt')
        pd.DataFrame(loss_hist).to_csv(self.exp_dir + '/errors__' + self.model.name + '__.csv', index=False)
        pd.DataFrame(loss_validate_hist).to_csv(self.exp_dir + '/errors_val__' + self.model.name + '__.csv', index=False)

        meta_data = {'n_epochs':n_epochs, 'lr':lr, 'batch_size': batch_size}
        with open(self.exp_dir + '/data.json', 'w') as fp:
            json.dump(meta_data, fp)


        return
