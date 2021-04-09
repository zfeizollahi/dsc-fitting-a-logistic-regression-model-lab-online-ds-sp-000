# Fitting a Logistic Regression Model - Lab

## Introduction

In the last lesson you were given a broad overview of logistic regression. This included an introduction to two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with `statsmodels`. For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the [Titanic](https://www.kaggle.com/c/titanic/data) shipwreck or not (yes, it's a bit morbid).


## Objectives

In this lab you will: 

* Implement logistic regression with `statsmodels` 
* Interpret the statistical results associated with model parameters

## Import the data

Import the data stored in the file `'titanic.csv'` and print the first five rows of the DataFrame to check its contents. 


```python
# Import the data
import pandas as pd

df = pd.read_csv('titanic.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Define independent and target variables

Your target variable is in the column `'Survived'`. A `0` indicates that the passenger didn't survive the shipwreck. Print the total number of people who didn't survive the shipwreck. How many people survived?


```python
# Total number of people who survived/didn't survive
df.Survived.sum()
```




    342



Only consider the columns specified in `relevant_columns` when building your model. The next step is to create dummy variables from categorical variables. Remember to drop the first level for each categorical column and make sure all the values are of type `float`: 


```python
# Create dummy variables
relevant_columns = ['Pclass', 'Age', 'SibSp', 'Fare', 'Sex', 'Embarked', 'Survived']
dummy_dataframe = pd.get_dummies(df[relevant_columns].dropna(), drop_first=True, dtype=float)

dummy_dataframe.shape
```




    (712, 8)



Did you notice above that the DataFrame contains missing values? To keep things simple, simply delete all rows with missing values. 

> NOTE: You can use the [`.dropna()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html) method to do this. 


```python
# Drop missing rows
dummy_dataframe.columns
```




    Index(['Pclass', 'Age', 'SibSp', 'Fare', 'Survived', 'Sex_male', 'Embarked_Q',
           'Embarked_S'],
          dtype='object')



Finally, assign the independent variables to `X` and the target variable to `y`: 


```python
# Split the data into X and y
y = dummy_dataframe['Survived']
X = dummy_dataframe.drop('Survived', axis=1)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Fare</th>
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>7.2500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>71.2833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Fit the model

Now with everything in place, you can build a logistic regression model using `statsmodels` (make sure you create an intercept term as we showed in the previous lesson).  

> Warning: Did you receive an error of the form "LinAlgError: Singular matrix"? This means that `statsmodels` was unable to fit the model due to certain linear algebra computational problems. Specifically, the matrix was not invertible due to not being full rank. In other words, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Build a logistic regression model using statsmodels
import statsmodels.api as sm
```

    /Users/zhaleh/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm



```python
X = sm.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()
```

    Optimization terminated successfully.
             Current function value: 0.444229
             Iterations 6


## Analyze results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
# Summary table
result.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   712</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   704</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     7</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 09 Apr 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.3417</td>  
</tr>
<tr>
  <th>Time:</th>                <td>12:08:26</td>     <th>  Log-Likelihood:    </th> <td> -316.29</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -480.45</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>5.360e-67</td>
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    5.6378</td> <td>    0.633</td> <td>    8.901</td> <td> 0.000</td> <td>    4.396</td> <td>    6.879</td>
</tr>
<tr>
  <th>Pclass</th>     <td>   -1.2102</td> <td>    0.163</td> <td>   -7.427</td> <td> 0.000</td> <td>   -1.530</td> <td>   -0.891</td>
</tr>
<tr>
  <th>Age</th>        <td>   -0.0433</td> <td>    0.008</td> <td>   -5.263</td> <td> 0.000</td> <td>   -0.059</td> <td>   -0.027</td>
</tr>
<tr>
  <th>SibSp</th>      <td>   -0.3796</td> <td>    0.125</td> <td>   -3.043</td> <td> 0.002</td> <td>   -0.624</td> <td>   -0.135</td>
</tr>
<tr>
  <th>Fare</th>       <td>    0.0012</td> <td>    0.002</td> <td>    0.474</td> <td> 0.635</td> <td>   -0.004</td> <td>    0.006</td>
</tr>
<tr>
  <th>Sex_male</th>   <td>   -2.6168</td> <td>    0.217</td> <td>  -12.040</td> <td> 0.000</td> <td>   -3.043</td> <td>   -2.191</td>
</tr>
<tr>
  <th>Embarked_Q</th> <td>   -0.8155</td> <td>    0.598</td> <td>   -1.363</td> <td> 0.173</td> <td>   -1.988</td> <td>    0.357</td>
</tr>
<tr>
  <th>Embarked_S</th> <td>   -0.4036</td> <td>    0.270</td> <td>   -1.494</td> <td> 0.135</td> <td>   -0.933</td> <td>    0.126</td>
</tr>
</table>



# Your comments here
Seems like, negative correlations with survival. The lower the class (meaning first class is "low") the more likely you survived, if you were female, more likely survived, then the younger you were, the more likely, followed by whether you had a sibling.

## Level up (Optional)

Create a new model, this time only using those features you determined were influential based on your analysis of the results above. How does this model perform?


```python
# Your code here
X = X[['Pclass', 'Age', 'SibSp', 'Sex_male', 'const']]
logit_model = sm.Logit(y, X)
result = logit_model.fit()
result.summary()
```

    Optimization terminated successfully.
             Current function value: 0.446755
             Iterations 6





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   712</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   707</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>     4</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 09 Apr 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.3379</td>  
</tr>
<tr>
  <th>Time:</th>                <td>12:13:05</td>     <th>  Log-Likelihood:    </th> <td> -318.09</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -480.45</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>5.015e-69</td>
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Pclass</th>   <td>   -1.3139</td> <td>    0.141</td> <td>   -9.324</td> <td> 0.000</td> <td>   -1.590</td> <td>   -1.038</td>
</tr>
<tr>
  <th>Age</th>      <td>   -0.0446</td> <td>    0.008</td> <td>   -5.457</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.029</td>
</tr>
<tr>
  <th>SibSp</th>    <td>   -0.3747</td> <td>    0.121</td> <td>   -3.098</td> <td> 0.002</td> <td>   -0.612</td> <td>   -0.138</td>
</tr>
<tr>
  <th>Sex_male</th> <td>   -2.6148</td> <td>    0.215</td> <td>  -12.177</td> <td> 0.000</td> <td>   -3.036</td> <td>   -2.194</td>
</tr>
<tr>
  <th>const</th>    <td>    5.5908</td> <td>    0.543</td> <td>   10.288</td> <td> 0.000</td> <td>    4.526</td> <td>    6.656</td>
</tr>
</table>



# Your comments here
Baseline seems to work better. I guess we'll learn about better methods later?

## Summary 

Well done! In this lab, you practiced using `statsmodels` to build a logistic regression model. You then interpreted the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Scikit-learn!
