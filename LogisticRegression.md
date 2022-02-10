# **Logistic Regression**

***Logistic Regression*** refers to a statistical model, in which, in its basic form, use **logistic function** to give output in form of binary variable. In regression analysis, **logisitc regression** refers to estimation of parameters of a logistic model, which is a form of binary model. In terms of mathematics, a binary model has mostly *one dependent variable*, which has binary outcomes, like, *pass/fail*, *true/false*, *yes/no*, etc.

## ***Mathematical Concept used***

Consider a model with *two independent numeric variables*, namely, ***x<sub>1</sub>*** and ***x<sub>2</sub>*** , and *one dependent (indictor) variable* ***p*** . The linear relationship between them based on *hypothesis* can be shown in below equation, where ***l*** refers to *log-odds* of ***p***.

![eq1](images/eq1.JPG)

*Here ***b*** is the base of *logarithmic* function, which can be understood as a hyper-parameter to the problem, and ***β<sub>1</sub>*** , ***β<sub>2</sub>*** and ***β<sub>3</sub>*** are parameters of logistic regression model.

By simple algebraic modification to the above equation, we get the new equation as below.

![eq2](images/eq2.JPG)

*Here ***S<sub>b</sub> (x)*** is defined as reciprocal of negative of *x* raised to the power of *b* .

## ***Interpretation***

The results of logistic model can be interpreted from the below equation.

![eq3](images/eq3.JPG)

*Here ***β<sub>0</sub>*** and ***β<sub>1</sub>*** are parameters of logistic model which are to be calculated and ***ε*** is the error term related to the estimated model.

The value of indicator variable will be **1** when the value of equation is greater than *0*, and iy will be **0** otherwise.

## ***Usage in Machine Learning***

Logistic Regression models the probability of the default class (base class, first class). Mathematically, it models that an input **X** belongs to the default class **Y=1** , which can be formally written as below.
## P(X) = P(Y=1 | X)
The parameters of the logistic regression algorithm must be estimated from your training data. This is done using maximum-likelihood estimation. Maximum-likelihood estimation is a common learning algorithm used by a variety of machine learning algorithms.

# Binomail Logistic Regression

<p>Binomial Logistic Regression mainly predicts the probability that an
observation falls into one of two categories of dichotomous dependent
variable based on one or more independent variables that can be either continues
or categorical! </p>

<h3>Explanation :</h3>

<p> Binomial Logistic Regression determines impact of multiple independent
variables presented together to predict membership of one or other
of two dependent variable categories. 
In general it can be told that it is used to predict relationship
between predictors and predicted variable.
predictors means independent variable.In dataset which features are used to predict.
There must be two or more independent variables(features)
or predictors for Binomial Logistic Regression.

</p>

<p> Mathematically we can say: <p>
<p> Let X £ <b>R<sup>d</sup></b> X=feature vector </p>
<p>Y £ {0,1}  [Y = label ]</p>
<p>X,  Y = random variables </p>
<p>The dependent variable Y is a nx1 vector

    So P(Y=1|X =x) = sigma(bita_0 + transpose(bita_1)*x)
Where sigma(u) = e<sup>u</sup> /(1+e<sup>u</sup>)
and bita_0 £ <b>R</b> and 
bita_1 £ <b> R <sup>d</sup></b> are parameters of model. 

</p>
<h3> Note :</h3>
<p> Logistic Regression is one of the binomial regression models and
uses logit as its link function! </p>

# Multiple logistic regression

## So,first of all multiple logistic regression is just an advance form of binary logistic regression or linear logistic regression.
As we know there are two category of dependent of outcome data in linear logistic regression but in multiple logistic
regression there are more than two category of dependent of outcome data  Like binary logistic regression, 
multinomial logistic regression uses maximum likelihood estimation to evaluate the probability of categorical relationship.
 
## For example:-
Lets say we have choose our stream after 10th class so there can be three types of dependent varibales that are:
1. Arts
2. Commerce
3. science
## As we can see from the categories that they are not ordered and there are no such possibilities of somone choosing one over
the other 

# Multiple logistic regression Vs Simple logistic regression

## The major and the most important difference in Multiple logistic regression and Simple logistic regression is that in Simple
logistic regression we only have to deal with two dependent categorical variables whereas in Multiple logistic regression
we have to deal with more than two types of dependent categorical variables.

# How to Apply multiple logistic regression 

## This is a very simple approach which include making k models for k classes as a set of independent binary regression
In this approach we are converting Mutli logistic regression into many simple logistic regressions models
Let's say there are three classes of outcomes as A,B and C and we to predict that what will be the outcome class
In first step we will create three models that are Class A vs rest, Class B vs rest and Class C vs rest
Which will be like in class A vs rest Class A will be 1 and rest will be zero like this
Then we will make equations of these three probabilities that are P(A),P(B) and P(C)
Now assign any record to the class based on input variables which has highest probability
Like if P(A)>P(B) and P(A)>P(C) then the output or outcome will be class A