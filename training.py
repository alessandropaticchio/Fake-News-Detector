from constants import *
import torch

def train(net, train_dl, valid_dl, epochs=100, early_stopping=True):
    net.train()

    # Init bad epochs counter for early stopping
    BAD_EPOCHS = 0
    BEST_VALID = 1 * 10 ^ 5
    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_dl, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        valid_loss, valid_accuracy = evaluate(net, valid_dl, criterion)
        train_accuracy = correct / total

        if valid_loss < BEST_VALID:
            BEST_VALID = valid_loss
            BAD_EPOCHS = 0
        else:
            BAD_EPOCHS += 1

        print('Epoch: {} | Train loss: {:.6f} | Train accuracy: {:.2f} | Val loss: {:6f} | Val accuracy: {:.2f}'.format(epoch,
                                                                                                          train_loss,
                                                                                                          train_accuracy,
                                                                                                          valid_loss,
                                                                                                          valid_accuracy))

        if BAD_EPOCHS == PATIENCE and early_stopping:
            break

    print('Done')
    return net

def evaluate(net, test_dl, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
    return correct/total, running_loss