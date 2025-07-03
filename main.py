import csv
import random
import math
import numpy as np

NUM_GAUSSIANS = 3
NUM_FEATURES = 2
PATH = "mog_samples.csv"
Xs = [] #inputs
Ys = [] #correct classes (just for checking accuracy)
EPOCHS = 10

with open(PATH, 'r') as file:
    reader =  csv.reader(file)
    next(reader)
    for line in reader:
        inp = list(map(float,line[:NUM_FEATURES]))
        cl = float(line[-1])
        Xs.append(inp)
        Ys.append(cl)

random.shuffle(Xs)
TEST_SIZE = 0.3
NUM_SAMPLES = len(Xs)
Xs_test,Ys_test = Xs[:int(NUM_SAMPLES*TEST_SIZE)], Ys[:int(NUM_SAMPLES*TEST_SIZE)]
Xs,Ys = Xs[int(NUM_SAMPLES*TEST_SIZE):], Ys[int(NUM_SAMPLES*TEST_SIZE):]
NUM_SAMPLES=int(NUM_SAMPLES*(1-TEST_SIZE))

def dist(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def av_lists(arr):
    # if len(arr)==0:
    #     return None
    total = [0 for i in range(len(arr[0]))]
    for i in arr:
        for ii in range(len(i)):
            total[ii] += i[ii]
    for i in range(len(total)):
        total[i]/=len(arr)
    return total

means = random.sample(Xs,NUM_GAUSSIANS)
current_group = [-1 for i in range(NUM_SAMPLES)]
while True: #initializing the gaussians using k means
    changes = 0
    groups = [[] for i in range(NUM_GAUSSIANS)]
    for _,i in enumerate(Xs):
        distances = [dist(i,ii) for ii in means]
        ii = distances.index(min(distances))
        groups[ii].append(i)
        if current_group[_]!=ii:
            changes+=1
        current_group[_]=ii

    for i in range(len(means)):
        new = av_lists(groups[i])
        means[i]=new

    if changes==0:
        break

def outer_prod(v1,v2):
    return [[v1[i]*v2[ii] for ii in range(len(v2))] for i in range(len(v1))]

def vector_op(v1,v2,op):
    assert len(v1)==len(v2), f"Invalid vector lengths {len(v1)} and {len(v2)}"
    return [eval(f"{v1[i]}{op}{v2[i]}") for i in range(len(v1))]

def vect_scale(v,scalar):
    return list(map(lambda x:x*scalar,v))

def mat_scale(matrix,scalar):
    height,width = len(matrix),len(matrix[0])
    for i in range(height):
        for ii in range(width):
            matrix[i][ii]*=scalar
    return matrix

def mat_op(m1,m2,op):
    assert len(m1)==len(m2) and len(m1[0])==len(m2[0]), f"Invalid vector lengths {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}"
    height,width = len(m1),len(m1[0])
    for i in range(height):
        for ii in range(width):
            m1[i][ii] = eval(f"{m1[i][ii]}{op}{m2[i][ii]}")
    return m1

weights = []
for i in range(NUM_SAMPLES):
    w = [0]*NUM_GAUSSIANS
    w[current_group[i]] = 1
    weights.append(w)

covariances = [None for i in range(NUM_GAUSSIANS)]
for i in range(NUM_GAUSSIANS):
    cov = [[0 for _ in range(NUM_FEATURES)] for _ in range(NUM_FEATURES)]
    sum_weights = sum(weights[_][i] for _ in range(NUM_SAMPLES))
    for ii in range(NUM_SAMPLES):
        weight = weights[ii][i]
        vect = vector_op(Xs[ii],means[i],'-')
        c = mat_scale(outer_prod(vect,vect),weight)
        cov = mat_op(cov,c,'+')
        # for j in range(NUM_FEATURES):
        #     cov[j][j] += 1e-3
    cov = mat_scale(cov,1/sum_weights)
    covariances[i]=cov

phi = [0]*NUM_GAUSSIANS
for i in range(NUM_GAUSSIANS):
    s = 0
    for ii in range(NUM_SAMPLES):
        s+=weights[ii][i]
    phi[i]=s/NUM_SAMPLES

def det(m): #Messy manual determinant calculator
    assert len(m)==len(m[0]), f"Not a square matrix {len(m)}x{len(m[0])}"
    for i in range(len(m)-1):
        for ii in range(i+1,len(m)):
            m[ii] = vector_op(m[ii],vect_scale(m[i],m[ii][i]/m[i][i]),'-')
    d = eval('*'.join([str(m[i][i]) for i in range(len(m))]))
    return d

def inverse(m): #I am NOT making an inverse calculator from scratch
    m = np.asarray(m)
    inv = np.linalg.inv(m)
    return inv.tolist()

def inner_prod(m1,m2): #inner product between matrices
    assert len(m1[0])==len(m2), f"Invalid matrix dimensions {len(m1)}x{len(m1[0])} and {len(m2)}x{len(m2[0])}"
    m2 = T(m2)
    h,w = len(m1), len(m2)
    product = [[0]*w]*h
    for i in range(w):
        for ii in range(h):
            v1 = m1[ii]
            v2 = m2[i]
            p = sum(v1[i]*v2[i] for i in range(len(v1)))
            product[ii][i]=p
    return product

def T(m): #returns transpose of a matrix
    rows = len(m)
    cols = len(m[0])
    transpose = [[m[i][ii] for i in range(rows)] for ii in range(cols)]
    return transpose

def gauss(x,mean,covariance):
    denom = (2*math.pi)**(NUM_FEATURES/2)*math.sqrt(det(covariance))
    v = vector_op(x,mean,'-')
    numer = math.exp(-0.5*inner_prod(inner_prod([v],inverse(covariance)), list(zip(v)))[0][0])
    return numer/denom


for i in range(EPOCHS):
    #E step
    for ii in range(NUM_SAMPLES):
        denom = 0
        for iii in range(NUM_GAUSSIANS):
            denom+=phi[iii]*gauss(Xs[ii],means[iii],covariances[iii])
        w = [phi[_]*gauss(Xs[ii],means[_],covariances[_])/denom for _ in range(NUM_GAUSSIANS)]
        weights[ii]=w

    #M step
    for i in range(NUM_GAUSSIANS):
        cov = [[0 for _ in range(NUM_FEATURES)] for _ in range(NUM_FEATURES)]
        sum_weights = sum(weights[_][i] for _ in range(NUM_SAMPLES))
        for ii in range(NUM_SAMPLES):
            weight = weights[ii][i]
            vect = vector_op(Xs[ii], means[i], '-')
            c = mat_scale(outer_prod(vect, vect), weight)
            cov = mat_op(cov, c, '+')
            # for j in range(NUM_FEATURES):
            #     cov[j][j] += 1e-3
        cov = mat_scale(cov, 1 / sum_weights)
        covariances[i] = cov


    for i in range(NUM_GAUSSIANS):
        s = 0
        for ii in range(NUM_SAMPLES):
            s += weights[ii][i]
        phi[i] = s / NUM_SAMPLES

    for i in range(NUM_GAUSSIANS):
        denom = sum(weights[_][i] for _ in range(NUM_SAMPLES))
        m = [0]*NUM_FEATURES
        for ii in range(NUM_SAMPLES):
            vector = vect_scale(Xs[ii],weights[ii][i])
            m = vector_op(m,vector,'+')
        m = vect_scale(m,1/denom)
        means[i]=m

loss = 0
for i in Xs_test:
    weight = [0]*NUM_GAUSSIANS
    denom = 0
    for ii in range(NUM_GAUSSIANS):
        denom += phi[ii] * gauss(i, means[ii], covariances[ii])
    weight = [phi[_] * gauss(i, means[_], covariances[_]) / denom for _ in range(NUM_GAUSSIANS)]

    prob = 0
    for ii in range(NUM_GAUSSIANS):
        prob += weight[ii]*gauss(i,means[ii],covariances[ii])
    loss+=math.log(prob)
loss/=-NUM_SAMPLES
print(f"Loss: {loss}")