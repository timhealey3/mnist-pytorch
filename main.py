from non_linear import NonLinearNN
from data_loader import training_data, test_data
from training import Training
from torch import nn
from NonLinearTorch import NonLinearTorch
from cnn import CNN

def main():
    trainingData = training_data()
    testData = test_data()
    nonLinearNN = NonLinearNN()
    nonLinearTraining = Training(nonLinearNN, trainingData, testData, 5, 16, nn.CrossEntropyLoss(), 0.01)
    nonLinearTraining.train()
    nonLinearTraining.test()
    nonLinearTorch = NonLinearTorch()
    nonLinearTorchTraining = Training(nonLinearTorch, trainingData, testData, 5, 16, nn.CrossEntropyLoss(), 0.01)
    nonLinearTorchTraining.train()
    nonLinearTorchTraining.test()
    cnn = CNN()
    cnnTraining = Training(cnn, trainingData, testData, 5, 16, nn.CrossEntropyLoss(), 0.01)
    cnnTraining.train()
    cnnTraining.test()
    

if __name__ == "__main__":
    main()