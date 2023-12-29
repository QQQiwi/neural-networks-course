import pandas as pd
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()



class NeuralNetwork:
    '''
    Нейронная сеть с:
        - 3 входами
        - скрытым слоем с 3 нейронами (h1, h2, h3)
        - выходной слой с 1 нейроном (o1)
    '''
    def __init__(self):
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
        # x входные элементы.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w7 * x[2] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.w8 * x[2] + self.b2)
        h3 = sigmoid(self.w9 * x[0] + self.w10 * x[1] + self.w11 * x[2] + self.b4)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.w12 * h3 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data - массив размерами (n x 2), где n -количество наблюдений в наборе.
        - all_y_trues - спиоск истинных значений (целевых).
        Элементы all_y_trues соответствуют наблюдениям в data.
        '''
        learn_rate = 0.1
        epochs = 1000 # сколько раз пройти по всему набору данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Прямой проход (эти значения нам понадобятся позже)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w7 * x[2] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.w8 * x[2] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w9 * x[0] + self.w10 * x[1] + self.w11 * x[2] + self.b4
                h3 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.w12 * h3 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Считаем частные производные.
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_w12 = h3 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                d_ypred_d_h3 = self.w12 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w7 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w8 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Нейрон h3
                d_h3_d_w9 = x[0] * deriv_sigmoid(sum_h3)
                d_h3_d_w10 = x[1] * deriv_sigmoid(sum_h3)
                d_h3_d_w11 = x[2] * deriv_sigmoid(sum_h3)
                d_h3_d_b4 = deriv_sigmoid(sum_h3)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w7
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w8
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон h3
                self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w9
                self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w10
                self.w11 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w11
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b4



                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.w12 -= learn_rate * d_L_d_ypred * d_ypred_d_w12
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

        # --- Считаем полные потери в конце каждой эпохи
        if epoch % 10 == 0:
            y_preds = pd.Series(list(map(self.feedforward, data)))
            loss = mse_loss(all_y_trues, y_preds)
            print("Эпоха %d потери: %.3f" % (epoch, loss))


if __name__ == "__main__":
    network = NeuralNetwork()
    data = [
        [-4, 1, 5],  # Наталья
        [7, -1, -4],   # Николай
        [4, 14, 10],   # Данил
        [-8, -18, 6], # Екатерина
    ]
    all_y_trues = [
        0, # Наталья
        1, # Николай
        1, # Данил
        0, # Екатерина
    ]
    network.train(data, all_y_trues)
