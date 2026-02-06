# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load the customer dataset and preprocess it by handling missing values and encoding categorical features.

### STEP 2: 
Split the dataset into training and testing sets to evaluate model performance.

### STEP 3: 
Define a neural network architecture with fully connected layers and ReLU activation functions.

### STEP 4: 
Select an appropriate loss function (CrossEntropyLoss) and optimizer (Adam) for multi-class classification.

### STEP 5: 
Train the neural network using the training data through forward pass, loss computation, and backpropagation.

### STEP 6: 
Test the trained model on unseen data and predict the customer segment (A, B, C, or D).

## PROGRAM

### Name:Adchayakiruthika M S 

### Register Number:212223230005

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
      for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

### Dataset Information
<img width="950" height="725" alt="Screenshot 2026-02-06 114337" src="https://github.com/user-attachments/assets/ba745171-ac46-4991-9003-d659e2c82a53" />

### OUTPUT

## Confusion Matrix
<img width="713" height="590" alt="image" src="https://github.com/user-attachments/assets/5b75c08b-f22d-450c-9877-805e0f842f99" />

## Classification Report
<img width="657" height="448" alt="image" src="https://github.com/user-attachments/assets/8a9aeb8d-05b4-48d9-9359-428987b65c5a" />

### New Sample Data Prediction
<img width="482" height="103" alt="image" src="https://github.com/user-attachments/assets/8c73e87e-8670-4951-8d9b-fd1742a4950e" />

## RESULT
A neural network classification model was successfully developed and trained to accurately predict customer segments (A, B, C, and D) for new market data.

