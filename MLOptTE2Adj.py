#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Create and evaluate forward models


# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[2]:


#read te data and split it
resmat = pd.read_excel("te_results_expanded.xlsx")
X = resmat.iloc[:,:4]*(10**9)
Y = resmat.iloc[:,11:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42);
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X, Y, random_state=None);
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(X, Y, random_state=None);
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(X, Y, random_state=None);
X_train_4, X_test_4, Y_train_4, Y_test_4 = train_test_split(X, Y, random_state=None);


# In[3]:


#scale train and test data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_test_1 = scaler.transform(X_test_1)
X_test_2 = scaler.transform(X_test_2)
X_test_3 = scaler.transform(X_test_3)
X_test_4 = scaler.transform(X_test_4)
r = np.amax(X_train, axis = 0) - np.amin(X_train, axis = 0)


# In[4]:


#train reference model - 3 hidden layers of 100 neurons each, 2000 iterations
nn_ref = MLPRegressor(solver="lbfgs", max_iter=2000, alpha=0.05, hidden_layer_sizes=(100,100,100))
nn_ref.fit(X_train, Y_train)


# In[5]:


#evaluate reference model
ref_score = [nn_ref.score(X_test, Y_test), nn_ref.score(X_test_1, Y_test_1), nn_ref.score(X_test_2, Y_test_2),
            nn_ref.score(X_test_3, Y_test_3), nn_ref.score(X_test_4, Y_test_4)]
print(np.average(ref_score))
print(ref_score)


# In[14]:


#creates a list of neuron layers build
def hls_list(min_l=2, max_l=5, step_l=1, min_n=100, max_n=500, step_n=100):
    hls = list()
    for layer in np.arange(min_l, max_l+1, step_l):
        for neuron in np.arange(min_n, max_n+1, step_n):
            tup = tuple()
            for i in range(layer):
                tup = tup + (neuron,)
            hls.append(tup)
    return hls


# In[15]:


#performing grid search to avoid overfitting - try 1
hls = hls_list(1,2,1,50,200,50)
m = [500, 1000, 1500, 2000]

par = {'hidden_layer_sizes':hls, 'max_iter':m}

nnl = MLPRegressor(solver="lbfgs", alpha=0.05)
regl_1 = GridSearchCV(nnl, par, scoring='r2', cv=5)
regl_1.fit(X_train,Y_train)
print("try 1:")
print(regl_1.cv_results_)
print(regl_1.best_estimator_)
print(regl_1.best_score_)
print(regl_1.best_params_)


# In[16]:


#performing grid search to avoid overfitting - try 2
hls = hls_list(2,3,1,200,300,50)
m = [1500, 2000, 2500]

par = {'hidden_layer_sizes':hls, 'max_iter':m}

nnl = MLPRegressor(solver="lbfgs", alpha=0.05)
regl_2 = GridSearchCV(nnl, par, scoring='r2', cv=5)
regl_2.fit(X_train,Y_train)
print("try 2:")
print(regl_2.cv_results_)
print(regl_2.best_estimator_)
print(regl_2.best_score_)
print(regl_2.best_params_)


# In[17]:


#performing grid search to avoid overfitting - try 3
hls = hls_list(3,5,1,150,250,50)
m = [2500, 3000, 3500]

par = {'hidden_layer_sizes':hls, 'max_iter':m}

nnl = MLPRegressor(solver="lbfgs", alpha=0.05)
regl_3 = GridSearchCV(nnl, par, scoring='r2', cv=5)
regl_3.fit(X_train,Y_train)
print("try 3:")
print(regl_3.cv_results_)
print(regl_3.best_estimator_)
print(regl_3.best_score_)
print(regl_3.best_params_)


# In[18]:


#performing grid search to avoid overfitting - try 4
hls = hls_list(4,6,1,100,200,50)
m = [2500]

par = {'hidden_layer_sizes':hls, 'max_iter':m}

nnl = MLPRegressor(solver="lbfgs", alpha=0.05)
regl_4 = GridSearchCV(nnl, par, scoring='r2', cv=5)
regl_4.fit(X_train,Y_train)
print("try 4:")
print(regl_4.cv_results_)
print(regl_4.best_estimator_)
print(regl_4.best_score_)
print(regl_4.best_params_)


# In[19]:


#performing grid search to avoid overfitting - try 5
hls = hls_list(6,8,1,100,200,50)
m = [2500]

par = {'hidden_layer_sizes':hls, 'max_iter':m}

nnl = MLPRegressor(solver="lbfgs", alpha=0.05)
regl_4 = GridSearchCV(nnl, par, scoring='r2', cv=5)
regl_4.fit(X_train,Y_train)
print("try 4:")
print(regl_4.cv_results_)
print(regl_4.best_estimator_)
print(regl_4.best_score_)
print(regl_4.best_params_)


# In[20]:


#Check best model
nn_gs_1 = regl_1.best_estimator_
nn_gs_2 = regl_2.best_estimator_
nn_gs_3 = regl_3.best_estimator_
nn_gs_4 = regl_4.best_estimator_
gs_1_score = [nn_gs_1.score(X_test, Y_test), nn_gs_1.score(X_test_1, Y_test_1), nn_gs_1.score(X_test_2, Y_test_2),
            nn_gs_1.score(X_test_3, Y_test_3), nn_gs_1.score(X_test_4, Y_test_4)]
gs_2_score = [nn_gs_2.score(X_test, Y_test), nn_gs_2.score(X_test_1, Y_test_1), nn_gs_2.score(X_test_2, Y_test_2),
            nn_gs_2.score(X_test_3, Y_test_3), nn_gs_2.score(X_test_4, Y_test_4)]
gs_3_score = [nn_gs_3.score(X_test, Y_test), nn_gs_3.score(X_test_1, Y_test_1), nn_gs_3.score(X_test_2, Y_test_2),
            nn_gs_3.score(X_test_3, Y_test_3), nn_gs_3.score(X_test_4, Y_test_4)]
gs_4_score = [nn_gs_4.score(X_test, Y_test), nn_gs_4.score(X_test_1, Y_test_1), nn_gs_4.score(X_test_2, Y_test_2),
            nn_gs_4.score(X_test_3, Y_test_3), nn_gs_4.score(X_test_4, Y_test_4)]
'''print(np.average(gs_1_score))
print(np.average(gs_2_score))
print(np.average(gs_3_score))'''
print(gs_1_score)
print(gs_2_score)
print(gs_3_score)
print(gs_4_score)


# In[ ]:


from matplotlib import pyplot
train_scores, test_scores = list(), list()
for al in a:
    nnl = MLPRegressor(solver="lbfgs", max_iter=2000, hidden_layer_sizes=al, alpha=0.05)
    nnl.fit(X_train,Y_train)
    # evaluate on the train dataset
    train_yhat = nnl.predict(X_train)
    train_acc = r2_score(Y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = nnl.predict(X_test)
    test_acc = r2_score(Y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (al[0], train_acc, test_acc))

# plot of train and test scores vs tree depth
pyplot.plot(b, train_scores, '-o', label='Train')
pyplot.plot(b, test_scores, '-o', label='Test')
pyplot.legend()
#pyplot.xscale('log')
pyplot.show()


# In[ ]:


print(scaler.inverse_transform(X_test), nn.predict(X_test))
print(scaler.inverse_transform(X_train), nn.predict(X_train), Y_train)


# In[ ]:


#Create gradient descent process and find optimal geometry


# In[6]:


#vector absolute value
def vecabs(a):
    s = 0
    for el in a:
        s = s + pow(el,2)
    return np.sqrt(s)


# In[ ]:


#loss calculation
def loss(s, est ,g, mu_g, rg):
    sp = abs(est.predict([g])[0])
    return (pow((sp[0]-s[0]),2) + pow((sp[1]-s[1]), 2))
''' + relu(abs(g[0]-mu_g[0]) - 1/2*rg[0]) + 
            relu(abs(g[1]-mu_g[1]) - 1/2*rg[1]) + relu(abs(g[2]-mu_g[2]) - 1/2*rg[2]) + 
            relu(abs(g[3]-mu_g[3]) - 1/2*rg[3]))'''


# In[8]:


#ReLU
def relu(x):
    return np.maximum(0,x)


# In[9]:


#calculate single iteration of gradient descent adjustment
def grad(s, est, g, delta, mu_g, rg):
    del0 = np.array([delta[0], 0, 0, 0])
    del1 = np.array([0, delta[1], 0, 0])
    del2 = np.array([0, 0, delta[2], 0])
    del3 = np.array([0, 0, 0, delta[3]])
    del_l0 = loss(s, est, g + del0, mu_g, rg) - loss(s, est, g, mu_g, rg)
    del_l1 = loss(s, est, g + del1, mu_g, rg) - loss(s, est, g, mu_g, rg)
    del_l2 = loss(s, est, g + del2, mu_g, rg) - loss(s, est, g, mu_g, rg)
    del_l3 = loss(s, est, g + del3, mu_g, rg) - loss(s, est, g, mu_g, rg)
    return [del_l0/delta[0], del_l1/delta[1], del_l2/delta[2], del_l3/delta[3]]


# In[10]:


#Find optimal g given initial g
def opt(s, est, g_ini, delta, lam, mu_g, rg, tol, max_iter):
    g_hat = g_ini
    prev = 0
    curr = loss(s, est, g_hat, mu_g, rg)
    count = 0
    while(abs(curr-prev) > tol and count < max_iter):
        prev = curr
        g_hat = g_hat - lam*np.array(grad(s, est, g_hat, delta, mu_g, rg))
        curr = loss(s, est, g_hat, mu_g, rg)
        count += 1
    print(count)
    return g_hat


# In[11]:


#Perform GD on ref model
nn = nn_ref
mu_g = np.mean(X_train, axis = 0)
delta = [1/10000, 1/10000, 1/10000, 1/2000]
g_min = X_train[0]
loss_min = 1
for x in X_train:
    g = opt([0,1], nn, x, delta, 0.01, mu_g, r, 1e-5, 1e3)
    l = loss([0,1], nn, g, mu_g, r)
    if (l < loss_min):
        g_min = g
        loss_min = l
print(loss_min)
nn.predict([g_min])


# In[12]:


#Present optimal reference geometry
g_min_ref = g_min
print("The optimal geometry, according to AI magic:", scaler.inverse_transform(g_min_ref))
print("The spectrum predicted by the model:", nn.predict([g_min_ref]))
nn_ref.predict([g_min_ref])


# In[13]:


#Compare with best measurement in tm train dataset - real loss = 0.001477
print("Optimal simulated geometry [x1, x2, gap, he]:", [100,50,50,700])
g0 = scaler.transform([[100,50,50,700]])
#print(g0[0])
sp = nn.predict(g0)[0]
ssim = [0.006251921, 0.962076716]
print("Simulated [p0, p1]:", ssim)
print("Predicted [p0, p1]:", sp)
ls = pow((ssim[0]-0),2) + pow((ssim[1]-1), 2)
lp = pow((sp[0]-0),2) + pow((sp[1]-1), 2)
print("Simulated loss:", ls)
print("Predicted loss:", lp)
if(ls>=lp):
    print("Is predicted better than simulated?: YES!")
else:
    print("Is predicted better than simulated?: NO!!")

#loss([0,1], nn, g0[0], mu_g, r)


# In[21]:


#Perform gradient descent for the smaller model
g_min = X_train[0]
loss_min = 1
for x in X_train:
    g = opt([0,1], nn_gs_4, x, delta, 0.01, mu_g, r, 1e-5, 1e3)
    l = loss([0,1], nn_gs_4, g, mu_g, r)
    if (l < loss_min):
        g_min = g
        loss_min = l
print(loss_min)
nn.predict([g_min])
#Present optimal CV geometry
g_min_ref = g_min
print("The optimal CV geometry, according to AI magic:", scaler.inverse_transform(g_min))
print("The spectrum predicted by the model:", nn.predict([g_min]))


# In[22]:


#Check gradient descent for 10000 tries - better?
nn = nn_ref
mu_g = np.mean(X_train, axis = 0)
delta = [1/10000, 1/10000, 1/10000, 1/2000]
g_min = X_train[0]
loss_min = 1
for x in X_train:
    g = opt([0,1], nn, x, delta, 0.01, mu_g, r, 1e-5, 1e4)
    l = loss([0,1], nn, g, mu_g, r)
    if (l < loss_min):
        g_min = g
        loss_min = l
print(loss_min)
nn.predict([g_min])
print("The optimal reference geometry, according to AI magic, 10000 tries:", scaler.inverse_transform(g_min))
print("The spectrum predicted by the model:", nn.predict([g_min]))


# In[23]:


#Check gradient descent for 10000 tries - better?
g_min = X_train[0]
loss_min = 1
for x in X_train:
    g = opt([0,1], nn_gs_4, x, delta, 0.01, mu_g, r, 1e-5, 1e4)
    l = loss([0,1], nn_gs_4, g, mu_g, r)
    if (l < loss_min):
        g_min = g
        loss_min = l
print(loss_min)
nn.predict([g_min])
#Present optimal CV geometry
g_min_ref = g_min
print("The optimal CV geometry, according to AI magic, 10000 tries:", scaler.inverse_transform(g_min))
print("The spectrum predicted by the model:", nn.predict([g_min]))


# In[ ]:




