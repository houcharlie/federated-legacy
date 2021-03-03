import numpy as np 
import matplotlib.pyplot as plt
d = 10
# some additional stuff
def gradient(A, b, x, y, l):
    gradx = np.dot(A,y) + l*x
    grady = np.dot(A.T,x) - y - b
    assert gradx.shape == (d,1)
    assert grady.shape == (d,1)
    return (gradx, grady)
def catalyst_gradient(A, b, x, y, l, x_, y_, tau):
    gradx = np.dot(A,y) + l*x + tau*(x - x_)
    grady = np.dot(A.T,x) - y - b - tau*(y - y_)
    assert gradx.shape == (d,1)
    assert grady.shape == (d,1)
    return (gradx, grady)
def func_value(A, b, x, y, l):
    return np.linalg.norm(np.dot(A, x) - b)**2/2 + l * np.linalg.norm(x)**2/2
# settings for plot 1
n = 10

R = 10000
K = 20
het1 = 10
het2 = 10
l = 0
eta = 0.01
eta_c = 0.01
#beta = 2
tau = 1

As_vals = np.random.normal(loc = 1, scale = het1, size = (n, d))
bs_vals = np.random.normal(loc = 1, scale = het2, size = (n, d))
average_b = np.mean(bs_vals, axis = 0)[np.newaxis]
#bs_vals = bs_vals - np.repeat(average_b, n, axis = 0) + np.ones((n,d))
bs_vals = bs_vals - np.repeat(average_b, n, axis = 0)
#print("b means", np.mean(bs_vals, axis = 0))
x = 10 * np.ones((d,1))
xmd = 10 * np.ones((d,1))
y = 10 * np.ones((d,1))
ymd = 10 * np.ones((d,1))
xs = []
ys = []
sc_xs = []
sc_ys = []
As = []
bs = []
catalyst_xs = []
catalyst_ys = []
for i in range(n):
    As.append(np.diag(As_vals[i,:]))
    bs.append(bs_vals[i,:][:,np.newaxis])
    xs.append(10*np.ones((d,1)))
    ys.append(10*np.ones((d,1)))
    sc_xs.append(10*np.ones((d,1)))
    sc_ys.append(10*np.ones((d,1)))
    catalyst_xs.append(10*np.ones((d,1)))
    catalyst_ys.append(10*np.ones((d,1)))
A = np.diag(np.mean(As_vals, axis = 0))
#A = np.eye(d)
#A = np.diag(np.array(np.linspace(1,beta, num=10)))
b = np.mean(bs_vals, axis = 0)[:,np.newaxis]
#b = np.mean(bs_vals, axis = 0)
#b = average_b
#solution = np.dot(np.linalg.pinv((np.dot(A.T, A) + l * np.eye(d)), rcond = 1e-20), b) 
#print(solution)
fedavg_x = xs[0]
fedavg_y = ys[0]
sc_x = sc_xs[0]
sc_y = sc_ys[0]
catalyst_x_ref = catalyst_xs[0]
catalyst_y_ref = catalyst_ys[0]
catalyst_x_var = catalyst_xs[0]
catalyst_y_var = catalyst_ys[0]

dist = []
fedavg_dist =[]
scaffold_dist = []
catalyst_dist = []
md_dist = []
counter = 0
last_func = func_value(A,b,catalyst_x_var, catalyst_y_var, l)
#solution = np.ones(d)
solution = np.zeros(d)
for r in range(R):
    if r % 100 == 0:
        print(r)
    (full_xgrad, full_ygrad) = gradient(A,b,sc_x, sc_y, l)
    (catalyst_fullx, catalyst_fully) = catalyst_gradient(A,b,catalyst_x_var, catalyst_y_var, l, catalyst_x_ref, catalyst_y_ref, tau)
    curr_func = func_value(A,b,catalyst_x_var, catalyst_y_var, l)
    if curr_func < last_func or counter > 2:
        catalyst_x_ref = catalyst_x_var 
        catalyst_y_ref = catalyst_y_var
        last_func = curr_func
        counter = 0
    else:
        counter += 1
    for i in range(n):
        for k in range(K):
            #FedAvg
            if k == 0 and i == 0:
                assert np.linalg.norm(xs[1] - xs[2]) < 1e-10
            (fedavg_gradx, fedavg_grady) = gradient(As[i], bs[i], xs[i], ys[i], l)
            x_ = xs[i] - eta * fedavg_gradx
            y_ = ys[i] + eta * fedavg_grady
            xs[i] = x_ 
            ys[i] = y_
            
            #SCAFFOLD
            (sc_xadj, sc_yadj) = gradient(As[i], bs[i], sc_x, sc_y, l)
            (sc_xgrad, sc_ygrad) = gradient(As[i], bs[i], sc_xs[i], sc_ys[i], l)
            x_ = sc_xs[i] - eta*(sc_xgrad - sc_xadj + full_xgrad)
            y_ = sc_ys[i] + eta*(sc_ygrad - sc_yadj + full_ygrad)
            sc_xs[i] = x_
            sc_ys[i] = y_

            #Catalyst
            (cata_xadj, cata_yadj) = catalyst_gradient(As[i],bs[i],catalyst_x_var, catalyst_y_var, l, catalyst_x_ref, catalyst_y_ref, tau)
            (cata_xgrad, cata_ygrad) = catalyst_gradient(A[i],bs[i],catalyst_xs[i], catalyst_ys[i], l, catalyst_x_ref, catalyst_y_ref, tau)
            x_ = catalyst_xs[i] - eta_c*(cata_xgrad - cata_xadj + catalyst_fullx)
            y_ = catalyst_ys[i] + eta_c*(cata_ygrad - cata_yadj + catalyst_fully)
            catalyst_xs[i] = x_ 
            catalyst_ys[i] = y_
    


    fedavg_x = np.mean(xs, axis = 0)
    fedavg_y = np.mean(ys, axis = 0)
    sc_x = np.mean(sc_xs, axis = 0)
    sc_y = np.mean(sc_ys, axis = 0)
    catalyst_x_var = np.mean(catalyst_xs, axis = 0)
    catalyst_y_var = np.mean(catalyst_ys, axis = 0)
    assert fedavg_x.shape == (d,1)
    assert fedavg_y.shape == (d,1)
    assert sc_x.shape == (d,1)
    assert sc_y.shape == (d,1)
    assert catalyst_x_var.shape == (d,1)
    assert catalyst_y_var.shape == (d,1)

    for i in range(n):
        xs[i] = fedavg_x
        ys[i] = fedavg_y
        sc_xs[i] = sc_x 
        sc_ys[i] = sc_y
        catalyst_xs[i] = catalyst_x_var
        catalyst_ys[i] = catalyst_y_var
    (xgrad, ygrad) = gradient(A, b, x, y, l)
    x_ = x - eta_c * xgrad 
    y_ = y + eta_c * ygrad
    (xgrad,ygrad) = gradient(A,b,x_,y_,l)
    x = x - eta_c*xgrad
    y = y + eta_c*ygrad

    (xgrad, ygrad) = gradient(A, b, xmd, ymd, l)
    x_ = xmd - eta/10 * xgrad 
    y_ = ymd + eta/10 * ygrad 
    xmd = x_ 
    ymd = y_
    '''
    dist.append(np.linalg.norm(x - solution))
    fedavg_dist.append(np.linalg.norm(fedavg_x - solution))
    scaffold_dist.append(np.linalg.norm(sc_x - solution))
    catalyst_dist.append(np.linalg.norm(catalyst_x_var- solution))
    '''
    dist.append(np.linalg.norm(x - solution))
    fedavg_dist.append(np.linalg.norm(fedavg_x - solution))
    scaffold_dist.append(np.linalg.norm(sc_x - solution))
    catalyst_dist.append(np.linalg.norm(catalyst_x_var - solution))
    md_dist.append(np.linalg.norm(xmd - solution))

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(range(R), md_dist, linestyle=':')
ax.plot(range(R), dist, linestyle='--')
ax.plot(range(R), fedavg_dist, linestyle='-')
ax.plot(range(R), scaffold_dist, linestyle='-.')
ax.plot(range(R), catalyst_dist, '-*')
ax.set_yscale('log')
plt.xlabel('rounds')
plt.ylabel(r'$|z - z^*|$')
plt.ylim(bottom = 1e-12)
plt.ylim(top = 1e3)
plt.legend(['Mirror Descent', 'Mirror-prox', 'FedAvg-S', 'SCAFFOLD-S', "SCAFFOLD-Catalyst-S"])
plt.savefig()
    

