"""

 * Name   : Perceptron-based-classification-algorithm - Machine Learning - Assignment #1
 * Created: 13 - 11 - 2018
 * Author :	Ahmed M. El-Shal
 
"""
# In[1]
"""Importing library"""
import glob
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("Done")

# In[2]
"""Loading the training data and create the X and Y training Matrices"""

X_train = []
load_img = glob.glob('Train/*.jpg')

for i in range(len(load_img)):
    load_img[i] = load_img[i].replace('Train/','')
    load_img[i] = load_img[i].replace('.jpg','')
    load_img[i] = int (load_img[i])
    
load_img = sorted(load_img)

for i in range(len(load_img)):
    load_img[i] = str (load_img[i])
    load_img[i] = "Train/"+ load_img[i]
    load_img[i] = load_img[i] + ".jpg"

for img in load_img:
    img = mpimg.imread(img)         
    img = np.reshape(img, 784)      
    img = np.append(img, 1)       
    img = img.reshape(len(img), 1)
    X_train.append(img)
print("traning data is loadded to X_train and it's shape is", np.shape(X_train))

# In[3]
"""Loading the Validation data and create the X and Y valid Matrices"""

X_valid = []
load_img = glob.glob('Validation/*.jpg')

for i in range(len(load_img)):
    load_img[i] = load_img[i].replace('Validation/','')
    load_img[i] = load_img[i].replace('.jpg','')
    load_img[i] = int (load_img[i])

load_img = sorted(load_img)         

for i in range(len(load_img)):
    load_img[i] = str (load_img[i])
    load_img[i] = "Validation/"+ load_img[i]
    load_img[i] = load_img[i] + ".jpg"

for img in load_img:
    img = mpimg.imread(img)        
    img = np.reshape(img, 784)      
    img = np.append(img, 1)         
    img = img.reshape(len(img), 1)
    X_valid.append(img)   

Y_valid =np.loadtxt('Validation/Validation Labels.txt')
print("Validation data is loadded to X_valid and it's shape is", np.shape(X_valid))
print("Y_valis shape is",np.shape(Y_valid))

# In[4]
"""Loading the testing data and create the X and Y test Matrices"""

X_test = []
load_img = glob.glob('Test/*.jpg')

for i in range(len(load_img)):
    load_img[i] = load_img[i].replace('Test/','')
    load_img[i] = load_img[i].replace('.jpg','')
    load_img[i] = int (load_img[i])

load_img = sorted(load_img)         

for i in range(len(load_img)):
    load_img[i] = str (load_img[i])
    load_img[i] = "Test/"+ load_img[i]
    load_img[i] = load_img[i] + ".jpg"

for img in load_img:
    img = mpimg.imread(img)     
    img = np.reshape(img, 784)  
    img = np.append(img, 1)     
    img = img.reshape(len(img), 1)
    X_test.append(img)   

Y_test =np.loadtxt('Test/Test Labels.txt')
print("Testing data is loadded to X_test and it's shape is", np.shape(X_test))   
print("Y_valis shape is",np.shape(Y_test))

# In[5]
"""Initialize the label fo the 10 classes"""

n_class =10
Y_train = []

for i in range(n_class):
    y_train = []
    i *= 240    
    z = i + 240 
    for j in range(len(X_train)):           
        if(i <= j < z):                 
            y_train.append(1)
        else:
            y_train.append(-1)
    Y_train.append(y_train)
print(np.shape(Y_train))

# In[6]
"""Define the traning model functions"""

def testPredection(x, w):
    return (np.dot(w.T , x))


def stepFunction(t):
    if t >= 0:
        return 1                   
    return -1                       
           
                            
def prediction(X, weight):
    return stepFunction(np.dot(weight.T, X))


def perceptronUpdate(X, y, weight, learningrate):
    error = 0        
    for i in range(len(X)):               
        y_hat = prediction(X[i],weight)   
        error += (y[i]-y_hat)**2           
        if ((y[i]-y_hat) == 2):            
                weight[:] = weight[:] + (X[i][:] * learningrate)
        elif ((y[i]-y_hat) == -2):
                weight[:] = weight[:] - (X[i][:] * learningrate)
    return weight, error


def trainPerceptronAlgorithm(X, y, weight, learningrate ):
    e = 1  
    n_iteration = 0
    while(e!=0):                      
        weight, e = perceptronUpdate(X, y, weight, learningrate)
        n_iteration+=1
    return weight, n_iteration
print("Done")

# In[7]
"""Intialize the learning rate"""

eta = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001,0.000001, 0.0000001, 0.00000001, 0.000000001]
print(len(eta))

# In[8]
"""Train the model with multiple learning rate"""

Weight=[]

for i in range(len(eta)):
    learn_rate = eta[i]
    W= []                                 
    for n in range(n_class):          
        w = []                                              
        for j in range(len(X_train[0])): 
            w.append(0.0)
        w[0] = 1.0
        w= np.asarray(w).reshape(len(w),1)  
        W.append(w)
        W[n], n_iterations = trainPerceptronAlgorithm(X_train,
                                                     Y_train[n], W[n], learn_rate)
        print ("Weight for class", n, "updated in ", n_iterations ,
                       "iterations forlearning rate =",learn_rate)
    Weight.append(W)                    
print(np.shape(Weight))
# In[9]
"""Test the trained model by using the testing data"""

Y_test1_predict = []
for i in range(len(eta)):
    Y_predict_for_eta = []
    for j in range(len(X_test)):         
        test1_out =[]                     
        for k in range(len(Weight[0])):      
            test1_out.append( testPredection(X_test[j], Weight[i][k]))
        Y_predict_for_eta.append(test1_out.index(np.max(test1_out)))
    Y_test1_predict.append(Y_predict_for_eta)
print(np.shape(Y_test1_predict))         

"""confusion matrix of testing data for 10 times of learning rate"""
t1_cm = []
t_acc =[]
y_true = Y_test
for i in range(len(eta)):
    y_pred = Y_test1_predict[i]
    t1_cm.append(confusion_matrix(y_true, y_pred))
    t_acc.append(accuracy_score(y_true, y_pred))
    print("Confusion Matrix for learning rate: ",eta[i])
    print(t1_cm)

for i in range(len(t1_cm)):
    name="Confusion Matrix " +str( i) +" Accuracy = " + str(t_acc[i])
    df_cm = pd.DataFrame(t1_cm[i], range(10), range(10))
    fig= plt.figure(figsize = (8, 6))
    fig.suptitle(t = name, x = 0.45, y = 0.95, fontsize = 20)
    sn.set(font_scale = 1)    #for label size
    sn.heatmap(df_cm, annot = True, annot_kws = {"size": 12})  

# In[10]
"""Test the trained model by using the validation data"""

Y_valid_predict = []
for i in range(len(eta)):
    Y_predict_for_eta = []
    for j in range(len(X_valid)):           
        valid_out =[]                      
        for k in range(len(Weight[0])):      
            valid_out.append( testPredection(X_valid[j], Weight[i][k]))
        Y_predict_for_eta.append(valid_out.index(np.max(valid_out)))
    Y_valid_predict.append(Y_predict_for_eta)
print(np.shape(Y_valid_predict))         
    
"""confusion matrix of validation data for 10 times of learning rate"""
v_cm = []
y_acc=[]
y_true = Y_valid
for i in range(len(eta)):
    y_pred = Y_valid_predict[i]
    v_cm.append(confusion_matrix(y_true, y_pred))
    y_acc.append( accuracy_score(y_true, y_pred))
    print(y_acc)
    print("Confusion Matrix for learning rate: ",eta[i])
    print(v_cm)

for i in range(len(v_cm)):
    name="Confusion Matrix " +str( i) +" Accuracy = " + str(y_acc[i])
    df_cm = pd.DataFrame(v_cm[i], range(10), range(10))
    fig= plt.figure(figsize = (8, 6))
    fig.suptitle(t = name, x = 0.45, y = 0.95, fontsize = 20)
    sn.set(font_scale = 1)    #for label size
    sn.heatmap(df_cm, annot = True, annot_kws = {"size": 12})  

# In[11]
"""geting the best 10 lines from the validation data"""

new_weight = []

for i in range((n_class)):
    values = []
    for j in range(len(v_cm)):   
        values.append(v_cm[j][i][i])
    print(values)
    f=values.index(np.max(values))
    if values[4] >= np.max(values):
        f= 4
    else:
        pass
    print(f ,"   ", i )
    new_weight.append(Weight[f][i])

print(np.shape(new_weight))

"""Test the testing data using the best validation lines of weights"""

Y_test2_predict = []
for i in range(len(X_test)):         
    test2_out =[]                     
    for j in range(len(new_weight)):       
        test2_out.append( testPredection(X_test[i], new_weight[j]))
    Y_test2_predict.append(test2_out.index(np.max(test2_out)))
print(np.shape(Y_test2_predict))   

"""confusion matrix of testing data for the best 10 lines"""
    
y_true = Y_test
y_pred = Y_test2_predict
t2_cm = (confusion_matrix(y_true, y_pred))
print(t2_cm)
print(np.shape(t2_cm))
acc = accuracy_score(y_true, y_pred)
print("acc", acc)
name="the best Learning Rate for each class \n in the Validation data generate a Test \n Confusion Matrix with Accuracy = " + str(acc)
df_cm = pd.DataFrame(t2_cm, range(10), range(10))
fig= plt.figure(figsize = (8, 6))
fig.suptitle(t = name, x = 0.45, y = 1.05, fontsize = 20)
sn.set(font_scale = 1)    #for label size
sn.heatmap(df_cm, annot = True, annot_kws = {"size": 12})  # font size


