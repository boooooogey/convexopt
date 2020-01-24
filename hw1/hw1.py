import cvxpy as cp
import numpy as np
import sys
from matplotlib import pyplot as plt

def prob1(X,y,C,name="prob1.png"):
    #optimization variables
    s = cp.Variable(X.shape[0])
    beta = cp.Variable(X.shape[1])
    beta0 = cp.Variable()
    #C = cp.Variable()
    #objective function
    obj = cp.Minimize(0.5*(cp.norm(beta)**2) + C*cp.sum(s))
    #constraints
    const = [ cp.multiply(y,X * beta + beta0) >= 1 - s, s >= 0 ]
    #problem
    prob = cp.Problem(obj,const)
    prob.solve()
    #print(beta.value)
    #print(prob.status)
    #plot
    Xp = X[np.where(y == 1)]
    Xn = X[np.where(y == -1)]
    xlin = np.linspace(-2,4,100)
    plt.scatter(Xp[:,0],Xp[:,1])
    plt.scatter(Xn[:,0],Xn[:,1])
    plt.plot(xlin,(-beta0.value - beta.value[0]*xlin) / beta.value[1])
    plt.savefig(name)
    plt.close()
    return prob, beta, beta0

def prob2(X,y,C):
    #Xtilda definition
    Xtilda = X * y.reshape((y.shape[0],-1))
    labelsn = np.where(y == -1)
    labelsp = np.where(y == 1)
    #plt.scatter(Xtilda[labelsn[0],0], Xtilda[labelsn[0],1])
    #plt.scatter(Xtilda[labelsp[0],0], Xtilda[labelsp[0],1])
    #plt.axvline(x=0)
    #plt.axhline(y=0)
    #plt.show()
    #Optimization Variables
    w = cp.Variable((Xtilda.shape[0]))
    #objective function
    obj = cp.Maximize(-0.5*cp.norm(np.transpose(Xtilda) * w)**2 + cp.sum(w))
    #constraints
    const = [0 <= w, w <= C*np.ones(Xtilda.shape[0]), w * y == 0]
    #problem
    prob = cp.Problem(obj,const)
    prob.solve()
    beta = np.dot(np.transpose(Xtilda),w.value)
    #print(np.dot(np.transpose(Xtilda),w.value))
    xlin = np.linspace(-3,3,100)
    #plt.axvline(x=0)
    #plt.axhline(y=0)
    #plt.scatter(X[labelsn[0],0], X[labelsn[0],1])
    #plt.scatter(X[labelsp[0],0], X[labelsp[0],1])
    #plt.plot(xlin,-1*xlin*beta[0]/beta[1])
    #plt.show()
    #ypred = np.dot(Xtilda,beta)
    #ypred[np.where(ypred > 0)] = 1
    #ypred[np.where(ypred < 0)] = -1
    #print(sum(ypred == y))
    return prob, beta 

def test_accuracy(beta, beta0, X, y):
    ypred = np.dot(X,beta.value) + beta0.value
    ypred[np.where(ypred >= 0)] = 1
    ypred[np.where(ypred < 0)] = -1
    return sum(ypred == y)/len(y)

def main():
    #read data
    d = np.genfromtxt(sys.argv[1], skip_header=0, delimiter=',')
    dtest = np.genfromtxt(sys.argv[2], skip_header = 0, delimiter=',')
    #x, y
    X = d[:,0:2]
    y = d[:,2]
    #xtest, ytest
    Xtest = dtest[:,0:2]
    ytest = dtest[:,2]
    #pad 1 to x for b0
    #X = np.append(X, np.ones((X.shape[0],1)),1)
    p1 = prob1(X,y,1,name="report.png")
    p2 = prob2(X,y,1)
    print(p1[0].value)
    #print(p2[0].value)
    #print(p2[1])
    #print(p2[1])
    accuracy = []
    for i in range(-5,6):
        p = prob1(X,y,pow(2,i),name="prob"+str(i)+".png")
        accuracy.append(1-test_accuracy(p[1],p[2],Xtest,ytest))
    plt.plot(range(-5,6),accuracy)
    plt.xlabel("C (log2 scaled)")
    plt.ylabel("Missclassification Error")
    plt.savefig("missVsC.png")
    plt.close()
    print(accuracy)
main()
