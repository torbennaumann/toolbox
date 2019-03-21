import sys
import torch
import pandas as pd

if __name__ == '__main__':
    # Get the total number of args passed to the prediction.py
    total_args = len(sys.argv)
    if total_args == 3:
        # Get the arguments list
        cmdargs = sys.argv
        model_file = cmdargs[1]
        data_file = cmdargs[2]

    else:
        print('ERROR: incorrect arguments!')
        print('predict.py <model_file> <data>')
        sys.exit(-1)

    dtype = torch.float
    device = torch.device("cpu")

    #
    # load data
    #

    #
    df = pd.read_csv(data_file)

    inputs = df.drop('target', axis=1)
    targets = df[['target']]

    x = torch.tensor(inputs.values, device=device, dtype=dtype)
    y = torch.tensor(targets.values, device=device, dtype=torch.long).squeeze()

    model = torch.load(model_file)

    outputs = model(x)
    y_pred = torch.max(outputs.data, 1)[1]

    misclassifiction = 1.0 * (y == y_pred).sum().item() / y_pred.size()[0]
    print('Accurracy: ', 100 * misclassifiction, '%')
