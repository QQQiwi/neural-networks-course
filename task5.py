import argparse
import numpy as np
import pandas as pd


def check_arg_type(arg):
    args = arg.split('=')
    if len(args) != 2:
        raise argparse.ArgumentTypeError('Аргументы введены некорректно.')
    return args[0], args[1]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='+', type=check_arg_type,
                        help='w=w.txt x=x.txt nn_output=nn.txt y=y.txt')
    params = dict(parser.parse_args().params)
    return params['w'], params['x'], params['nn'], params['y'], params['epochs'], params['loss']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((np.array(y_true) - np.array(y_pred)) ** 2).mean()


class HiddenLayer:
    w = []
    inp = []
    out = []
    m = 0
    n = 0
    def __init__(self, w):
        self.w = w
        self.m = len(w[0])
        self.n = len(w)
        self.inp = []
        self.out = [0 for _ in range(self.n)]
    

class NeuralNetwork:
    layers_amount = 0
    layers = []
    # biases = []
    learning_rate = 0.1
    train_data = []
    true_values = []
    epoch_amount = 0
    dd = []
    loss_path = ""

    def __init__(self, w, x, y, epochs, loss_path):
        self.layers = [HiddenLayer(cur_w) for cur_w in w]
        self.layers_amount = len(self.layers)
        self.train_data = x
        self.true_values = y
        self.epoch_amount = epochs
        self.dd = [[]] * self.layers_amount
        self.loss_path = loss_path


    def feedforward(self):
        for i in range(self.layers_amount):
            cur_layer = self.layers[i]
            cur_layer_y = np.dot(cur_layer.w, cur_layer.inp)
            for j in range(cur_layer.n):
                cur_layer_y[j] = sigmoid(cur_layer_y[j])
            cur_layer.out = cur_layer_y.copy()
            if i < self.layers_amount - 1:
                self.layers[i + 1].inp = cur_layer_y.copy()
        return self.layers[-1].out.tolist()
    

    def update_weights(self):
        for p in range(self.layers_amount):
            cur_layer = self.layers[p]
            for i in range(cur_layer.m):
                for j in range(cur_layer.n):
                    cur_layer.w[i][j] -= self.learning_rate * self.dd[p][i] * cur_layer.inp[j]


    def train(self):
        loss = []
        output = ""
        for cur_epoch in range(self.epoch_amount):
            predicted_values = []    
            for x, y_true in zip(self.train_data, self.true_values):
                self.layers[0].inp = x

                y = self.feedforward()
                predicted_values.append(y)

                y_amount = len(y)
                self.dd[-1] = [0 for _ in range(y_amount)]
                for i in range(y_amount):
                    self.dd[-1][i] = (y[i] - y_true[i]) * deriv_sigmoid(y[i])
                
                for p in range(self.layers_amount - 1, 0, -1):
                    cur_layer = self.layers[p]
                    self.dd[p - 1] = [0 for _ in range(y_amount)]
                    for i in range(cur_layer.n):
                        for j in range(cur_layer.m):
                            self.dd[p - 1][i] += cur_layer.w[j][i] * self.dd[p][j]
                        self.dd[p - 1][i] *= deriv_sigmoid(self.layers[p - 1].out[i])
                
                self.update_weights()

            if cur_epoch % 10:
                loss_value = mse_loss(self.true_values, predicted_values)
                output += f"Ошибка на эпохе {cur_epoch} равна {loss_value}\n"
                loss.append(loss_value)

        with open(self.loss_path, 'w', encoding='utf-8') as file:
            file.write(output)


    def save(self, save_path):
        with open(save_path, 'w', encoding='utf-8') as file:
            weights = [layer.w for layer in self.layers]
            file.write(str(weights))


def get_list_data_from_file(data_path):
    with open(data_path, 'r') as file:
        data_list = file.read()
    data_list = eval(data_list)
    return data_list


def get_random_weights(m, n):
    return np.random.rand(m, m, n).tolist()


def main():
    try:
        w_path, x_path, nn_output, y_path, epochs, loss_path = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0

    x = get_list_data_from_file(x_path)
    y = get_list_data_from_file(y_path)

    if w_path == "None":
        w = get_random_weights(len(x[0]), len(x[0]))
    else:
        w = get_list_data_from_file(w_path)

    network = NeuralNetwork(w, x, y, int(epochs), loss_path)
    network.train()
    network.save(nn_output)


if __name__ == "__main__":
    main()

