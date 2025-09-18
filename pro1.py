import numpy as np

# creation of fake dataset with noise
np.random.seed(int(input("enter number: ")))
x = np.random.rand(100,1)
y =  6 * x + 7 + np.random.randn(100,1) * 0.1

# weights,bias
wt = np.random.randn(1)
b = np.random.randn(1)
learn_rate = 0.1
train_loop = 1000

# training loop - gradient descent
for loop in range(train_loop):
    # prediction
    y_pred=x.dot(wt)+b
    
    # MSE
    loss=np.mean((y - y_pred) ** 2)
    
    # Gradients
    dw = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # new weights
    wt_new = wt - learn_rate * dw
    b_new = b - learn_rate * db
    
    # steps
    if loop % 100 == 0:
        print(f"Loop {loop}, Loss: {loss:.4}, Weight: {wt[0]:.4f}, b:{b[0]:.4f}")  

# final
print("Trained model: y = {:.2f}x + {:.2f}".format(wt[0], b[0]))