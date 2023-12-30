import argparse
import numpy as np


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
    return params['w'], params['x'], params['nn'], params['y']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        self.out = []
    

class NeuralNetwork:
    layers_amount = 0
    layers = []
    # biases = []
    outputs = []

    def __init__(self, w, x):
        self.layers = [HiddenLayer(cur_w) for cur_w in w]
        self.layers[0].inp = x
        self.outputs = [0 for _ in range(len(x))]
        self.layers_amount = len(self.layers)


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
        w_path, x_path, nn_output, y_path = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0

    x = get_list_data_from_file(x_path)
    if w_path == "None":
        w = get_random_weights(len(x), len(x))
    else:
        w = get_list_data_from_file(w_path)

    network = NeuralNetwork(w, x)
    y = network.feedforward()
    with open(y_path, 'w', encoding='utf-8') as file:
        file.write(str(y))
    network.save(nn_output)


if __name__ == "__main__":
    main()

