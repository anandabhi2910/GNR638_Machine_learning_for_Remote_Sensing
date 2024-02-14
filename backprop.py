import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights and biases
np.random.seed(42)  # for reproducibility
w_h_1 = np.random.randn(2)  # weights for hidden layer
w_h_2=np.random.randn(2)
w_o_1 = np.random.randn(2)  # weights for output layer
w_o_2=np.random.randn(2)
b_h = np.random.randn(2)  # biases for hidden layer, adjusted to be a row vector/'
b_o = np.random.randn(2)  # bias for output layer

# Adjust the input x to have shape (1, 2)
x = np.array([1,1])

# Target output
target_y = np.array([0,0])

# Training loop
epochs = 10
learning_rate = 0.1

for epoch in range(epochs):
    # Forward Pass
    # Hidden layer
    z_h_1 = np.dot(x, w_h_1) + b_h[0]
    z_h_2=np.dot(x,w_h_2)+b_h[1]
    a_h = sigmoid(np.array([z_h_1,z_h_2]))
    # Output layer
    z_o_1 = np.dot(a_h, w_o_1) + b_o[0]
    z_o_2=np.dot(a_h,w_o_2)+b_o[1]
    a_o=sigmoid(np.array([z_o_1,z_o_2]))
    # print(z_o.shape)
    y_pred = a_o
    print("y_pred_shape:", y_pred.shape)
    print("shape_target_y:", target_y.shape)
    # Calculate Loss
    loss = np.linalg.norm(y_pred - target_y)
    print(y_pred - target_y)

    # Backward Pass
    # Output layer gradients
    d_w3=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)*a_h
    d_w4=-2*(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)*a_h
    d_b3=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)
    d_b4=-2*(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)
    # Hidden layer gradients
    d_w1=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)*w_o_1[0]*sigmoid_derivative(z_h_1)*x-2(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)*w_o_2[0]*sigmoid_derivative(z_h_1)*x
    d_w2=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)*w_o_1[1]*sigmoid_derivative(z_h_2)*x-2(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)*w_o_2[1]*sigmoid_derivative(z_h_2)*x
    d_b1=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)*w_o_1[0]*sigmoid_derivative(z_h_1)-2(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)*w_o_2[0]*sigmoid_derivative(z_h_1)
    d_b2=-2*(target_y[0]-a_o[0])*sigmoid_derivative(z_o_1)*w_o_1[1]*sigmoid_derivative(z_h_2)-2(target_y[1]-a_o[1])*sigmoid_derivative(z_o_2)*w_o_2[1]*sigmoid_derivative(z_h_2)
    # Update Weights and Biases
   
    w_h_1=w_h_1-learning_rate*d_w1
    w_h_2=w_h_2-learning_rate*d_w2
    b_h[0]=b_h[0]-learning_rate*d_b1
    b_h[1]=b_h[1]-learning_rate*d_b2
    w_o_1=w_o_1-learning_rate*d_w3
    w_o_2=w_o_2-learning_rate*d_w4

    # Print gradients computed through backpropagation
    print(f"Epoch {epoch + 1}:")
    print("Gradients through backpropagation:")
    print("d_w1:", d_w1)
    print("d_w2:", d_w2)
    print("d_w3:", d_w3)
    print("d_w4:", d_w4)
    print('d_b1',d_b1)
    print('d_b2',d_b2)
    print('d_b3',d_b3)
    print('d_b4',d_b4)

    # Analytically compute gradients using the limit definition
    eps = 1e-4  # small value for numerical stability
    eps_1=[eps,0]
    eps_2=[0,eps]
    #forward pass ep1 wh1
    z_h_1_ep1 = np.dot(x, w_h_1+eps_1) + b_h[0]
    z_h_2_ep1=np.dot(x,w_h_2)+b_h[1]
    a_h_ep1 = sigmoid(np.array([z_h_1_ep1,z_h_2_ep1]))
    z_o_1_ep1 = np.dot(a_h_ep1, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h_ep1,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))
    #forward pass ep2 wh1
    z_h_1_ep2 = np.dot(x, w_h_1+eps_2) + b_h[0]
    z_h_2_ep2=np.dot(x,w_h_2)+b_h[1]
    a_h_ep2 = sigmoid(np.array([z_h_1_ep2,z_h_2_ep2]))
    z_o_1_ep2 = np.dot(a_h_ep2, w_o_1) + b_o[0]
    z_o_2_ep2=np.dot(a_h_ep2,w_o_2)+b_o[1]
    a_o_ep2=sigmoid(np.array([z_o_1_ep2,z_o_2_ep2]))

    grad_w1 = [(np.linalg.norm(target_y - a_o_ep1)-loss)/ eps,(np.linalg.norm(target_y-a_o_ep2)-loss)/eps]

    #forward pass ep1 wh2
    z_h_1_ep1 = np.dot(x, w_h_1) + b_h[0]
    z_h_2_ep1=np.dot(x,w_h_2+eps_1)+b_h[1]
    a_h_ep1 = sigmoid(np.array([z_h_1_ep1,z_h_2_ep1]))
    z_o_1_ep1 = np.dot(a_h_ep1, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h_ep1,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))

    #forward pass ep2 wh2
    z_h_1_ep2 = np.dot(x, w_h_1) + b_h[0]
    z_h_2_ep2=np.dot(x,w_h_2+eps_2)+b_h[1]
    a_h_ep2 = sigmoid(np.array([z_h_1_ep2,z_h_2_ep2]))
    z_o_1_ep2 = np.dot(a_h_ep2, w_o_1) + b_o[0]
    z_o_2_ep2=np.dot(a_h_ep2,w_o_2)+b_o[1]
    a_o_ep2=sigmoid(np.array([z_o_1_ep2,z_o_2_ep2]))
    
    grad_w2 = [(np.linalg.norm(target_y - a_o_ep1)-loss)/ eps,(np.linalg.norm(target_y-a_o_ep2)-loss)/eps]

    #forward pass ep1 wo1
    z_o_1_ep1 = np.dot(a_h, w_o_1+eps_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))
    #forward pass ep2 wo1
    z_o_1_ep2 = np.dot(a_h, w_o_1+eps_2) + b_o[0]
    z_o_2_ep2=np.dot(a_h,w_o_2)+b_o[1]
    a_o_ep2=sigmoid(np.array([z_o_1_ep2,z_o_2_ep2]))

    grad_w3=[(np.linalg.norm(target_y - a_o_ep1)-loss)/ eps,(np.linalg.norm(target_y-a_o_ep2)-loss)/eps]

    #forward pass ep1 wo1
    z_o_1_ep1 = np.dot(a_h, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h,w_o_2+eps_1)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))
    #forward pass ep2 wo1
    z_o_1_ep2 = np.dot(a_h, w_o_1) + b_o[0]
    z_o_2_ep2=np.dot(a_h,w_o_2+eps_2)+b_o[1]
    a_o_ep2=sigmoid(np.array([z_o_1_ep2,z_o_2_ep2]))

    grad_w4=[(np.linalg.norm(target_y - a_o_ep1)-loss)/ eps,(np.linalg.norm(target_y-a_o_ep2)-loss)/eps]

    #forward pass b1
    z_h_1_ep1 = np.dot(x, w_h_1) + b_h[0]+eps
    z_h_2_ep1=np.dot(x,w_h_2)+b_h[1]
    a_h_ep1 = sigmoid(np.array([z_h_1_ep1,z_h_2_ep1]))
    z_o_1_ep1 = np.dot(a_h_ep1, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h_ep1,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))

    grad_b1=[(np.linalg.norm(target_y-a_o_ep1)-loss)/eps]

    #forward pass b2
    z_h_1_ep1 = np.dot(x, w_h_1) + b_h[0]
    z_h_2_ep1=np.dot(x,w_h_2)+b_h[1]+eps
    a_h_ep1 = sigmoid(np.array([z_h_1_ep1,z_h_2_ep1]))
    z_o_1_ep1 = np.dot(a_h_ep1, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h_ep1,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))

    grad_b2=[(np.linalg.norm(target_y-a_o_ep1)-loss)/eps]

    #forward pass b3
    z_o_1_ep1 = np.dot(a_h, w_o_1) + b_o[0]+eps
    z_o_2_ep1=np.dot(a_h,w_o_2)+b_o[1]
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))

    grad_b3=[(np.linalg.norm(target_y-a_o_ep1)-loss)/eps]
    
    #forward pass b4
    z_o_1_ep1 = np.dot(a_h, w_o_1) + b_o[0]
    z_o_2_ep1=np.dot(a_h,w_o_2)+b_o[1]+eps
    a_o_ep1=sigmoid(np.array([z_o_1_ep1,z_o_2_ep1]))

    grad_b4=[(np.linalg.norm(target_y-a_o_ep1)-loss)/eps]

    print("Gradients analytically computed:")
    print("grad_w1:", grad_w1)
    print("grad_w2:", grad_w2)
    print("grad_w3:", grad_w3)
    print("grad_w4:",grad_w4)
    print('grad_b1',grad_b1)
    print('grad_b2',grad_b2)
    print('grad_b3',grad_b3)
    print('grad_b4',grad_b4)
    print("loss",loss)