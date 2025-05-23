from non_linear import NonLinearNN
from data_loader import training_data, test_data
from training import Training
from torch import nn
from NonLinearTorch import NonLinearTorch

def main():
    trainingData = training_data()
    testData = test_data()
    nonLinearNN = NonLinearNN()
    training = Training(nonLinearNN, trainingData, testData, 5, 16, nn.CrossEntropyLoss(), 0.01)
    training.train()
    training.test()
    nonLinearTorch = NonLinearTorch()
    training = Training(nonLinearTorch, trainingData, testData, 5, 16, nn.CrossEntropyLoss(), 0.01)
    training.train()
    training.test()

if __name__ == "__main__":
    main()