import pytest
import numpy as np
from experiment import NeuralNetwork

def test_neural_network():
    """测试神经网络是否能正确学习 XOR 问题"""
    nn = NeuralNetwork([2, 2, 1])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    
    flag = 0
    for i in X:
        pred = nn.predict(i)
        if 0.4 <= pred <= 0.6:  # 允许一定误差范围
            flag += 1
    
    assert flag == 4, "神经网络未能正确学习 XOR 问题"