# Support Vector Machine
## Introduction to SVM
Support Vector Machine also known as SVM in short is a supervised machine learning algorithm. It is used for regression as well as classification problem but commonly used for classification problems. It uses the concept of hyperplane. It can be a line, 2D planes or even n-dimentional planes. We will learn more about hyperplane in later section.<br>
The advantage or pros of SVM is that it works well with high dimensional space but the disadvantage of SVM is, it takes a lot of time to for training and therefore it is not advisable to use when the dataset is too large. 

> Fun fact: Although SVM can deal with non-linear data but the algorithm belongs to Linear Machine Learning Models. 

## Applications of SVM
Support Vector Machine algorithm is used in many fields some of them are:-
1.  Image recognition 
2.  Face detection 
3.  Voice detection
4.  Handwriting recognition

## Important concepts related to SVM
[![SVM 2d](https://github.com/snozh5/temp/blob/main/SVM%20pic/SVM%202D%20plane.png?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/SVM%202D%20plane.png)

- **Hyperplane:-**  It is a boundary or decision plane which categorize the dataset for example classifies Spam emails from the ham ones.
- **Support vectors:-** The data points that are nearest to the hyperplane which helps it to give the best possible boundary, hence supporting the hyperplane so they are called as Support Vector. 
- **Margin:-** The gap or the perpendicular distance between the line and support vector is called as margin. The best fit margin is the one with the maximum gap. 

## Types of SVM and Working Principle
Support Vector Machine can be divided into two types:-
1. **Linear SVM:-** In linear SVM the dataset can be divided using just a single straight line. Meaning dataset is linearly separable and categorize into two classes.  
[![Linear SVM](https://github.com/snozh5/temp/blob/main/SVM%20pic/Linear%20SVM.png?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/Linear%20SVM.png)

In order to understand the ***working*** of ***linear SVM*** let us take an example of a dataset where we need to identify for spam and ham emails. The standard equation of a line is given by `ax + by + c = 0`. We can generalize the equation `W0 + W1x1 + W2x2=0`, where 'x1' and 'x2' are the features — such as 'word_freq_technology' and 'word_freq_offer' — and W1 and W2 are the coefficients. If we substitute the value of x1 and x2 which are features in the equation of any given line with W coefficients, it will return a value.  
A ***positive value*** (blue points in the plot above) simply mean that the values are in one class; however, a ***negative value*** (red points in the plot above) would mean that is of the other class. The point lies on the line i.e. on hyperplane if the value is zero because any point on the line will satisfy the equation: `W0 + W1x1 + W2x2=0`.
> 3rd line(Hyperplane) should be considered as best fit classifier in the above figure. 
2. **Non-linear SVM:-** In non-linear SVM the dataset cannot be divided simply by drawing a straight line. Meaning the dataset in not linearly separable and cannot be categorized into two classes by just fitting a straight line which was the case of a linear SVM.

[![Non linear SVM](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM.PNG)

For the working principle of non linear SVM consider the image above. The data points are non linear and it cannot be separated by just fitting a straight line. So in order to separate these data points a new dimension needs to be introduced. For linear data we got x and y as dimension, so the new dimension is z and the formula is `z=x^2 +y^2`. So after adding the 3rd dimension SVM will divide the data points into different classes which will look like in the figure below:

[![Non linear SVM 3D](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM%203D.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM%203D.PNG)

If we take z=1 we can view it in 2d space which will look like the figure below:

[![Non linear SVM 2D](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM%202D.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/Non%20linear%20SVM%202D.PNG)

Thus the data points got classified into two classes and that's how non-linear SVM works. 

## Kernels 

 Kernels are functions which help to transform non-linear datasets. They take input as a low dimension and transform it into a high dimension space and such technique is called as kernel trick. Given a dataset, we can try various kernels, and choose the one that produces the best model. Top three commonly used kernel functions are:-

- **The Linear Kernel:-** Ths kernel is the basic of all, as the name suggest it linearly separate the classes. Mostly used for text recognition problems. Computational time is faster than other kernels. 
- **The Polynomial Kernel:-** This is kernel is used to create polynomial decision boundaries in non-linear datasets.
- **The Radial Basis Function (RBF) kernel:-** This kernel is one of the most popular and largely used when the dataset is non-linear. It has the ability to convert highly non-linear feature to linear space.

### Implementation
The implementation is done using python that uses a package name [sklearn](https://scikit-learn.org/stable/modules/svm.html) through which we need to import SVM.
```sh
#This tutorial will help you to learn to choose the best keranls
#types of SVM - 1.linear SVM, 2.kernal SVM
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.datasets import make_circles
 
#create a dataset 
training_dataset, value = make_circles(n_samples=500,noise=.05,factor=.5)
 
#plot dataset 
plt.scatter(training_dataset[:,0],training_dataset[:,1],c=value) 
plt.show()
```
[![Dataset 1](https://github.com/snozh5/temp/blob/main/SVM%20pic/dataset1.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/dataset1.PNG)

```sh
#three types of kernals 
kernal=['linear','poly','rbf']
```
```sh
#train and predict for each kernel
for kernal in kernals:
    clf=svm.SVC(kernel=kernal)
    #train
    clf.fit(training_dataset,value)
    
    #test
    prediction =clf.predict([[-0.75,-0.75]])
    print(f'SVC with {kernal} kernal:\nprediction for [-0.75,-0.75] :{prediction}')
   
    #plot the line, points
    X=training_dataset
    y=value
    X0=X[np.where(y==0)]
    X1=X[np.where(y==1)]

    plt.figure()
    x_min = X[:,0].min()
    x_max = X[:,0].max()
    y_min = X[:,1].min()
    y_max = X[:,1].max()

    XX,YY=np.mgrid[x_min:x_max:200j,y_min:y_max:200j]

    Z=clf.decision_function(np.c_[XX.ravel(),YY.ravel()])

    #put the result into a color plot
    Z=Z.reshape(XX.shape)

    plt.pcolormesh(XX,YY,Z>0,
                  cmap=plt.cm.Paired)

    plt.contour(XX,YY,Z, colors=['k','k','k'],
               linestyles=['--','-','--'],
               levels=[-.5,0,.5])

    plt.scatter(X0[:,0],X0[:,1],c='r',s=50)
    plt.scatter(X1[:,0],X1[:,1],c='b',s=50)

    plt.show()
```
[![linear Kernel](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultL.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultL.PNG)<br>
[![Polynomial Kernel](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultP.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultP.PNG)<br>
[![RBF Kernel](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultR.PNG?raw=true)](https://github.com/snozh5/temp/blob/main/SVM%20pic/resultR.PNG)






