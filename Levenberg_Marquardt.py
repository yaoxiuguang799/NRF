import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random
random.seed(0)

# 线性函数模型
def polynomialModel(params,x,q):
    def polynomial2(params,x):
        a = params[0,0]
        b = params[1,0]
        c = params[2,0]
        return a + b * x + c * pow(x, 2)
    def polynomial3(params,x):
        a = params[0,0]
        b = params[1,0]
        c = params[2,0]
        d = params[3,0]
        return a + b * x + c * pow(x, 2) + d * pow(x, 3)
    def polynomial4(params,x):
        a = params[0,0]
        b = params[1,0]
        c = params[2,0]
        d = params[3,0]
        e = params[4,0]
        return a + b * x + c * pow(x, 2) + d * pow(x, 3) + e * pow(x, 4)
    def polynomial5(params,x):
        a = params[0,0]
        b = params[1,0]
        c = params[2,0]
        d = params[3,0]
        e = params[4,0]
        f = params[5,0]
        return a + b * x + c * pow(x, 2) + d * pow(x, 3) + e * pow(x, 4) + f * pow(x, 5)
    if q == 2:
        return polynomial2(params,x)
    elif q == 3:
        return polynomial3(params,x)
    elif q == 4:
        return polynomial4(params,x)
    elif q == 5:
        return polynomial5(params,x)

# 线性函数模型：y=a+b*x. 需要求解的参数为a,b
def linearModel(params,x):
    a = params[0,0]
    b = params[1,0]
    return a + b * x

# AVC模型函数模型：y=Acos(w*x+θ)+b. 需要求解的参数为A,theta,b
def avcModel(params,x):
    A = params[0,0]
    theta = params[1,0]
    b = params[2,0]
    w = 2*np.pi/365
    return b + A*np.cos(w * x + theta)

# AVCE模型函数模型：y=Acos(w*x+θ)+b+lamda*DPWVera5. 需要求解的参数为A,theta,b
def avceModel(params,x):
    A = params[0,0]
    theta = params[1,0]
    b = params[2,0]
    lamda = params[3,0]
    w = 2*np.pi/365
    d = x[:,0].reshape(len(x[:,0]),1)
    DPWVera5 = x[:,1].reshape(len(x[:,0]),1)
    return b + A*np.cos(w * d + theta) + lamda*DPWVera5

def Levenberg_Marquardt(x,y,xk,callback):
    n_obs = len(y)
    n_params = xk.shape[0]
    J = mat(np.zeros((n_obs,n_params)))      #雅克比矩阵
    fx = mat(np.zeros((n_obs,1)))     # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((n_obs,1)))

    lase_mse = 0
    step = 0
    u,v= 1,2
    conve = 10000  # 最大迭代次数
    xk_l = []  # 用来存放每次迭代的结果
    def Deriv(params,x,n_obs):  # 对函数求偏导
        x1 = params.copy()
        x2 = params.copy()
        x1[n_obs,0] -= 0.000001
        x2[n_obs,0] += 0.000001
        p1 = callback(x1,x)
        p2 = callback(x2,x)
        d = (p2-p1)*1.0/(0.000002)
        return d
    while conve:
        
        mse,mse_tmp = 0,0
        step += 1  
        fx = callback(xk,x) - y
        mse += sum(fx**2)
        for j in range(n_params): 
            J[:,j] = Deriv(xk,x,j) # 数值求导                                                    
        mse /= n_obs  # 范围约束
    
        H = J.T*J + u*np.eye(n_params)   # n_params*n_params
        dx = -H.I * J.T*fx        # 
        xk_tmp = xk.copy()
        xk_tmp += dx
        fx_tmp =  callback(xk_tmp,x) - y  
        mse_tmp = sum(fx_tmp[:,0]**2)
        mse_tmp /= n_obs
        #判断是否下降
        q = float((mse - mse_tmp)/((0.5*dx.T*(u*dx - J.T*fx))[0,0]))
        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
    
            if s > temp:
                u = u*s
            else:
                u = u*temp
        else:
            u = u*v
            v = 2*v
            xk = xk_tmp
        xk_l.append(xk)
        # print ("step = %d,abs(mse-lase_mse) = %.8f" %(step,abs(mse-lase_mse)))  
        if abs(mse-lase_mse)<0.000001:
            break
        
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
        
    return xk_l

def test_avcModel():
    n = 365   
    A,theta,b = 10, 2.6, 14.4 # 这个是需要拟合的函数y(x) 的真实参数
    h0 = np.arange(1,366,1).reshape(n,1) # 横坐标，天
    y0 = [b + A*np.cos((2*np.pi/365) * t + theta)+random.gauss(0,2) for t in h0] # 带有噪声的序列
    h = np.arange(1,366,1).reshape(n,1)
    y = [b + A*np.cos((2*np.pi/365) * t + theta)+random.gauss(0,2) for t in h]
    # del_ind = np.array([60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,80,87,88,89,90,50,51,52,53,54,55,56,57])
    # del_ind = np.array([50,51,52,53,54,55,56,57,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,80,87,88,89,90,93,94,95,96,97,98])
    del_ind1 = np.arange(20,55,1)
    del_ind2 = np.arange(60,230,1)
    del_ind3 = np.arange(235,360,1)
    del_ind = np.concatenate((del_ind1,del_ind2,del_ind3))
    h = np.delete(h,del_ind)
    y = np.delete(y,del_ind)
    n=len(h)
    h = h.reshape((n,1))
    y = y.reshape((n,1))

    params0 = mat([[8.0],[2.0],[12.0]])
    xk_l = Levenberg_Marquardt(h,y,params0,avcModel)

    #用拟合好的参数画图
    plt.figure(figsize=(10, 6))
    plt.title(f'Optimization Results (iter {len(xk_l)})', fontsize=30)
    plt.scatter(h, y, s=30, c='b')
    plt.scatter(h0, y0, s=30, c='r')
    for i in range(len(xk_l)):
        if i == 1 or i == 2 or i == 4 or i == len(xk_l)-1:
            xk_i = xk_l[i]
            z = [avcModel(xk_i, j) for j in h0]
            plt.plot(h0, z, label=f'Iter {i+1}', linewidth=3)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.show()

def test_linearModel():
    n = 365   
    a,b = 5,10 # 这个是需要拟合的函数y(x) 的真实参数
    h0 = np.arange(1,366,1).reshape(n,1) # 横坐标，天
    y0 = [b + a*t +random.gauss(0,50) for t in h0] # 带有噪声的序列
    h = np.arange(1,366,1).reshape(n,1)
    y = [b + a*t +random.gauss(0,50) for t in h0] # 带有噪声的序列
    del_ind1 = np.arange(20,55,1)
    del_ind2 = np.arange(60,230,1)
    del_ind3 = np.arange(235,360,1)
    del_ind = np.concatenate((del_ind1,del_ind2,del_ind3))
    h = np.delete(h,del_ind)
    y = np.delete(y,del_ind)
    n=len(h)
    h = h.reshape((n,1))
    y = y.reshape((n,1))

    params0 = mat([[1.0],[0.0]])
    xk_l = Levenberg_Marquardt(h,y,params0,linearModel)

    #用拟合好的参数画图
    plt.figure(figsize=(10, 6))
    plt.title(f'Optimization Results (iter {len(xk_l)})', fontsize=30)
    plt.scatter(h, y, s=30, c='b')
    plt.scatter(h0, y0, s=30, c='r')
    for i in range(len(xk_l)):
        if i == 1 or i == 2 or i == 4 or i == len(xk_l)-1:
            xk_i = xk_l[i]
            z = [linearModel(xk_i, j) for j in h0]
            plt.plot(h0, z, label=f'Iter {i+1}', linewidth=3)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.show()

def test_polynomialModel():
    n = 365   
    a,b,c,d = 10,5,3,2 # 这个是需要拟合的函数y(x) 的真实参数
    h0 = np.arange(1,366,1).reshape(n,1) # 横坐标，天
    y0 = [a + b*t + c*t*t + d*t*t*t + random.gauss(0,50) for t in h0] # 带有噪声的序列
    h = np.arange(1,366,1).reshape(n,1)
    y = [a + b*t + c*t*t + d*t*t*t + random.gauss(0,50) for t in h0] # 带有噪声的序列
    del_ind1 = np.arange(20,55,1)
    del_ind2 = np.arange(60,230,1)
    del_ind3 = np.arange(235,360,1)
    del_ind = np.concatenate((del_ind1,del_ind2,del_ind3))
    h = np.delete(h,del_ind)
    y = np.delete(y,del_ind)
    n=len(h)
    h = h.reshape((n,1))
    y = y.reshape((n,1))

    params0 = mat([[0.0],[1.0],[1.0],[1.0]])
    polynomial3 = polynomialModel(params0,h,3)
    xk_l = Levenberg_Marquardt(h,y,params0,polynomial3)

    #用拟合好的参数画图
    plt.figure(figsize=(10, 6))
    plt.title(f'Optimization Results (iter {len(xk_l)})', fontsize=30)
    plt.scatter(h, y, s=30, c='b')
    plt.scatter(h0, y0, s=30, c='r')
    for i in range(len(xk_l)):
        if i == 1 or i == 2 or i == 4 or i == len(xk_l)-1:
            xk_i = xk_l[i]
            z = [linearModel(xk_i, j) for j in h0]
            plt.plot(h0, z, label=f'Iter {i+1}', linewidth=3)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=20)
    plt.show()
if __name__ == '__main__':
    test_polynomialModel()