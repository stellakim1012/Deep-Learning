# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# 학습데이터
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# 리스트를 넘파이 배열로 변경
x_data = np.array(x)
y_data = np.array(y)

# 파라미터 및 하이퍼 파라미터 설정
w, b, lr, epochs = 0, 0, 0.03, 2001

# 배치 경사 하강법 실행
for i in range(epochs):
    y_hat = w * x_data + b  # 추정
    error = y_hat - y_data  # 오차
    #w_diff = (1/len(x_data)) * sum(x_data * error)
    #b_diff = (1/len(x_data)) * sum(error)
    w_diff = np.mean(x_data * error)
    b_diff = np.mean(error)
    w = w - lr * w_diff
    b = b - lr * b_diff
    if i % 100 == 0:
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, w, b))

# 그래프 출력
y_pred = w * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()