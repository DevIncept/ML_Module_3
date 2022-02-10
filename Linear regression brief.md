 <h2 style="color:green" align="center">  Linear Regression </h2>

#### ➡Linear Regression is a machine learning algorithm based on supervised learning.

#### ➡It performs a regression task.

#### ➡Regression models a target prediction value based on independent variables.

#### ➡It is mostly used for finding out the relationship between variables and forecasting.

#### ➡Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.

<h3 style="color:purple">Sample problem of predicting home prices</h3>

Below table represents current home prices in monroe township based on square feet area

<img src="homepricetable.JPG" style="width:370px;height:250px">

**Problem Statement**: Given above data build a machine learning model that can predict home prices based on square feet area

# Hypothesis function for Linear Regression :

# y = B0 + B1*x

While training the model we are given :
x: input training data (univariate – one input variable(parameter))
y: labels to data (supervised learning)

When training the model – it fits the best line to predict the value of y for a given value of x. The model gets the best regression fit line by finding the best B0 and B2 values.
B0: intercept
B1: coefficient of x



# How to update B1 and B2 values to get the best fit line ?

Cost Function (J):
By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the B1 and B2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).

Cost function(J) of Linear Regression is the Root Mean Squared Error (RMSE) between predicted y value (pred) and true y value (y).

# Gradient Descent

To update B1 and B2 values in order to reduce Cost function (minimizing RMSE value) and achieving the best fit line the model uses Gradient Descent. The idea is to start with random B1 and B2 values and then iteratively updating the values, reaching minimum cost.


You can represent values in above table as a scatter plot (values are shown in red markers). After that one can draw a straight line that best fits values on chart. 

<img src="scatterplot.JPG" style="width:600px;height:370px">

You can draw multiple lines like this but we choose the one where total sum of error is minimum

<img src="equation.PNG" style="width:600px;height:370px" >

You might remember about linear equation from your high school days math class. Home prices can be presented as following equation,

home price = m * (area) + b

Generic form of same equation is,




 # Y = a + bX
