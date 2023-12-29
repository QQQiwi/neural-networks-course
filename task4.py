import pandas as pd
import math
import argparse


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
    return params['w'], params['x'], params['nn_output'], params['y']


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


class NeuralNetwork:
    '''
    Нейронная сеть с:
        - 3 входами
        - скрытым слоем с 3 нейронами (h1, h2, h3)
        - выходной слой с 1 нейроном (o1)
    '''
    weights = []

    layers_amount = 3

    def __init__(self, w):
        # Веса
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.w5 = 1
        self.w6 = 1
        self.w7 = 1
        self.w8 = 1
        self.w9 = 1
        self.w10 = 1
        self.w11 = 1
        self.w12 = 1
        # Смещения
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w7 * x[2] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.w8 * x[2] + self.b2)
        h3 = sigmoid(self.w9 * x[0] + self.w10 * x[1] + self.w11 * x[2] + self.b4)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.w12 * h3 + self.b3)
        return o1
    

    def save(save_path):
        pass


def get_list_data_from_file(data_path):
    pass


def get_random_weights(m, n):
    pass


def main():
    try:
        w_path, x_path, nn_output, y_path = parse_arguments()
    except:
        print("Ошибка чтения аргументов!")
        return 0

    x = get_list_data_from_file(x_path)
    if w_path == "None":
        w = get_random_weights(NeuralNetwork.layers_amount, len(x))
    else:
        w = get_list_data_from_file(w_path)

    network = NeuralNetwork(w)
    y = network.feedforward(x)
    with open(y_path, 'w', encoding='utf-8') as file:
        file.write(str(y))
    network.save(nn_output)


if __name__ == "__main__":
    main()

