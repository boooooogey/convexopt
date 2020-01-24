import numpy as np
from matplotlib import pyplot as plt
import sys

def soft(b,lamb,mask):
    new_b = np.copy(b)
    for i in range(len(set(mask))-1):
        indices = np.where(mask == (i+1))
        norm_b = np.linalg.norm(new_b[indices])
        if norm_b <= lamb:
            new_b[indices] = 0
        else:
            new_b[indices] = new_b[indices] - lamb / norm_b * np.sqrt(len(indices[0])) * new_b[indices]
    return new_b

def h(beta, lamb, mask):
    weight_of_each_group = []
    for i in range(len(set(mask))-1):
        indices = np.where(mask == (i+1))
        weight_of_each_group.append(np.sqrt(len(indices[0])) * np.linalg.norm(beta[indices]))
    return lamb * np.sum(weight_of_each_group), weight_of_each_group

def g(beta, X, y):
    return - np.dot(y, np.dot(X, beta)) + np.sum(np.log(1+np.exp(np.dot(X, beta))))

def total_loss(beta, X, y, lamb, mask):
    return g(beta, X, y) + h(beta, lamb, mask)[0]

def gradient_g(beta, X, y):
    return -np.dot(np.transpose(X), y) + np.dot(np.transpose(X), np.exp(np.dot(X, beta))/(1+np.exp(np.dot(X,beta))))

def step(beta, X, y, t, lamb, mask):
    nabla = gradient_g(beta, X, y)
    return soft(beta - t * nabla, lamb * t, mask)

def backtracking(b, X, y, lamb, mask, beta, alpha):
    t = 1
    i = 0
    def G(x, t_x):
        return (x - soft(x - t_x * gradient_g(x, X, y), t_x * lamb, mask))/t_x
    while True:
        i += 1
        if g(b - t * G(b, t),X,y) > g(b, X, y) - t * np.dot(gradient_g(b, X, y), G(b, t)) + t*alpha*np.dot(G(b,t),G(b,t)):
            t = t * beta
        else:
            break
    return t, i

def training_loop_backtracking(beta, X, y, lamb, mask, number_of_iterations):
    beta_plus = beta
    loss = [total_loss(beta, X, y, lamb, mask)]
    ts = []
    total_number_iteration = 0
    all_iterations = [0]
    for i in range(number_of_iterations):
        total_number_iteration += 1
        t, inner_iterations = backtracking(beta_plus, X, y, lamb, mask, 0.1, 0.5)
        total_number_iteration += inner_iterations
        all_iterations.append(total_number_iteration)
        ts.append(t)
        #print(t)
        beta_plus = step(beta_plus, X, y , t, lamb, mask)
        #print(beta_plus)
        #print(total_loss(beta_plus, X, y, lamb, mask))
        loss.append(total_loss(beta_plus, X, y, lamb, mask))
    return beta_plus, np.array(loss), ts, all_iterations 

def training_loop(beta, X, y, t, lamb, mask, number_of_iterations):
    beta_plus = beta
    loss = [total_loss(beta, X, y, lamb, mask)]
    for i in range(number_of_iterations):
        beta_plus = step(beta_plus, X, y , t, lamb, mask)
        loss.append(total_loss(beta_plus, X, y, lamb, mask))
    return beta_plus, np.array(loss)

def training_loop_nesterov(beta, X, y, t, lamb, mask, number_of_iterations):
    beta_k = np.copy(beta)
    beta_k1 = np.copy(beta)
    v = np.copy(beta)
    loss = [total_loss(beta, X, y, lamb, mask)]
    for i in range(number_of_iterations):
        if i > 1:
            v = beta_k + (i-2)/(i+1) * (beta_k - beta_k1) 
        else:
            v = beta_k
        beta_k1 = np.copy(beta_k)
        beta_k = step(v, X, y , t, lamb, mask)
        loss.append(total_loss(beta_k, X, y, lamb, mask))
    return beta_k, np.array(loss)

def predict(X, beta):
    out = np.dot(X, beta)
    y = np.array(out >= 0, dtype = np.int)
    return y

def accuracy(y, ypred):
    return np.mean(y == ypred)

def main():
    train_data = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    train_label = np.genfromtxt(sys.argv[2], skip_header=0, delimiter=',')
    group_label = np.append(0,np.genfromtxt(sys.argv[3], skip_header=0, delimiter=','))
    train_data = np.append(np.ones((train_data.shape[0],1)),train_data,1)
    t = 1e-4
    lamb = 5
    f_optimal = 336.207
    b = np.random.rand(len(group_label))
    print(backtracking(b,train_data, train_label, lamb, group_label, 0.8, 0.5))
    print(backtracking(b,train_data, train_label, lamb, group_label, 0.8, 0.001))
    exit()
    final_b, losses = training_loop(b, train_data, train_label, t, lamb, group_label, 1000)
    final_b_nesterov, losses_nesterov = training_loop_nesterov(b, train_data, train_label, t, lamb, group_label, 1000)
    final_b_backtracking, losses_backtracking, ts, all_number_of_iterations = training_loop_backtracking(b, train_data, train_label, lamb, group_label,400)
    plt.plot(range(len(losses)), np.log(losses-f_optimal),label="proximal gradient descent")
    plt.plot(range(len(losses_nesterov)),np.log(losses_nesterov-f_optimal), label="nesterov accelerated")
    plt.plot(all_number_of_iterations,np.log(losses_backtracking-f_optimal), label="proximal gradient descent w/backtracking")
    plt.ylabel("Loss (Log Scaled)")
    plt.xlabel("Step")
    plt.legend()
    plt.show()
    test_data = np.genfromtxt(sys.argv[4], skip_header=0, delimiter=',')
    test_data = np.append(np.ones((test_data.shape[0],1)),test_data,1)
    test_label = np.genfromtxt(sys.argv[5], skip_header=0, delimiter=',')
    print(accuracy(predict(test_data, final_b_nesterov),test_label))
    group_weights = h(final_b_nesterov, lamb, group_label)[1]
    ii = np.argmax(group_weights)
    print(str(ii+1) + "/" + str(len(group_weights)))

main()
