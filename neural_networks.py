# -*- coding: utf-8 -*-
######################################################################################
#####   IMPLEMENTING NEURAL   NETWORKS
# Predicting medical costs( Prédire les frais médicaux. ): 
# First 1: loading the data
"""
Predicting medical costs:  loading the data

Every machine learning pipeline starts with data and a task. Let's take a look at the
"Medical Cost Personal Datasets dataset", which consists of seven columns with the following descriptions:

|---------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------|
|                 Column Names                |                   Description                                             |                       Data type                                    |
|---------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------|
|                                             |                                                                           |                                                                    |
|                      age                    |                     age of primary beneficiary                            |                  numerical / integer                               |
|                                             |                                                                           |                                                                    |
|                      sex                    |                    insurance contractor gender                            |           ( integer female is 1, male is 0 )                       |
|                                             |                                                                           |                                                                    |
|                      bmi                    |                    body  mass index                                       |                  numerical / real value                            |
|                                             |                                                                           |                                                                    |
|                      children               |                    number of children    covered by health insurance      |                        numerical integer                           | 
|                                             |                                                                           |                                                                    |
|                      smoker                 |                    smoking or not                                         |           true is 1, false is 0                                    |
|                                             |                                                                           |                                                                    |
|                   region                    |                  the beneficiary's residential area in the US             |          categorical (northeast, northwest, southeast, southwest)  |
|                                             |                                                                           |                                                                    | 
|                  charges                    |                individual medical costs billed by health insurance        |             numerical/real value                                   |
|_____________________________________________|___________________________________________________________________________|____________________________________________________________________|

----------------------------------------------------------
"""
"""
We would like to predict the individual medical costs (charges) 
given the rest of the columns/features.
 Since charges represent continuous values (in dollars), 
 we’re performing a regression task. Think about the potential implications
  of using sex or bmi to predict what the individual insurance charges should be? 
  Should they be even used for prediction?
"""
## Our data is in the .csv format and we load it pandas:
"""
dataset = pd.read_csv('insurance.csv') 
"""
# view the first 5 entries of the dataset
"""
print(dataset.head())
"""
# Next we split the data into features (independent variables)
# and the target variable ( dependent variable )

# dataframe slicing using iloc
"""
features = dataset.iloc[:,0:6]
"""
# we select the last column with -1
"""
labels = dataset.iloc[:,-1]
"""
"""
The pandas shape property tells us the shape of our data --- a vector of two
values: the number of samples and the number of features.To check the shape of our dataset,
we can do :
"""
# print(features.shape)
"""
Or, to make things clearer:
"""
# print("Number of features:", features.shape[1])
# print("Number of samples:", features.shape[0])

"""
To see a useful summary statistics pf the dataset we do:
"""
# print(features.describe())
#=====================================================================================================
"""
                                                IMPLEMENTING NEURAL NETWORKS

"""
# Data preprocessing: one-hot encoding and standardization
"""One-hot encoding of categorical features:
"""
# Since neural networks cannot work with string data directly,we need to convert our categorical features
# ("region") into numerical.
"""
One-hot encoding creates a binary column for each category.For example, 
since the "region" variable has four categories, the one-hot encoding will result in four binary column:
"""
# northeast

# northwest

# southeast as shown in the table below.

"""
__________________________________
|   age | .......  |  region     |
|-------|----------|-------------|
|   32 | .......   |    NE       |
|-------|----------|-------------|  
|   35 | .......   |     SW      |
|-------|----------|-------------|
|   ... | .......  |     ...     |
|-------|----------|-------------|
________________________________________________________
|   age | .......  |... |  NE    |  NW  |  SE  |  SW    |
|-------|----------|----|--------|------|------|--------|
|   32  | .......  | ...| 1      |  0   |   0  |   0    |
|-------|----------|----|--------|------|------|--------|
|   35  | .......  | ...|   0    |   0  |   0  |   1    |
|-------|----------|----|--------|------|------|--------|
                      |   |_________________________|
                      |                         |
                      |______one-hot encoding___|                     

"""
# One-hot encoding can be accomplished by using 
"""pandas """
#   get_dummies() function:
"""
features = pd.get_dummies(features)
"""
#######################################################
# Split data into train and test sets:
"""
In machine learning, we train a model on a training data, and
 we evalute its performance on a
"""#held-out
"""set of data, our test set, not seen during the learning:"""
#########################

# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train,
# labels_test = train_test_split(features #the data
# , labels # the vector of the setdata
# , test_size=0.33 # Here we chose the test size to be 33% of the total data
# , random_state=42)
"""
Here we chose the test size to be 33% of the total data(test_size=0.33),
 and random state (random_state=42) controls the shuffling applied  to the data before applying the split.

"""
#================================================================
"""Standardize/normalize numerical features:
"""
"""Standardize"""
# The usual preprocessing step for numerical variables, 
# among others, is standardization that rescales features to Zero
# mean and unit variance.Why do we want to do that?
# Well, our features have different scales or units:"age" has an interval
# of [18, 64] and the "children" column's interval is much smaller, [0, 5].
# By having features with differing scales, the optimizer might update some 
# weights faster than the others.
"""Normalization"""
# Normalization is another way of preprocessing numerical data:it scales the numerical features to a fixed range - 
# usually between 0 and 1

# So which should you use? Well, there isn't always a clear answer,but you can try them all out and choose the one method that gives the best results.

# To normalize the numerical features we use an exciting addition to scikit-learn, columnTransformer, in the following way:

"""
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer 
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('normalize', Normalizer(),
['age', 'bmi', 'children'])], 
remainder = 'passthrough')

features_train = ct.fit_transform(features_train)

features_test = ct.transform(features_test)
"""
"""
The name of the column transformer is "only numeric", it applies a Normalizer()
to the 'age', 'bmi' , and 'children columns, and for the rest of the columns it just passes
through.
ColumnTransformer() returns'
"""
# NumPy
"""
arrays and we convert them back to a pandas DataFrame so we can see some
useful summaries of the scaled data.
"""
# To convert a numPy array back into a pandas DataFrame, we can do:
"""
features_train_norm = pd.DataFrame(features_train_norm,
columns = features_train.columns)
"""
# Note that we fit the scaler to the training data only, and them we apply the trained scaler onto the test data.
# This way we avoid "information leakage" from the training set to the test set.
# These two datasets should be completely unaware of each other!









# install pandas on your machine to use c:\users\abdo\appdata\local\programs\python\python39\python.exe -m pip install pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
# install tensorflow with the command below ==>
# python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def neural_network():
  """IMPLEMENTING NEURAL NETWORKS
  """
  print('Data preprocessing: one-hot encoding and standardization')
  #load the dataset
  dataset = pd.read_csv('insurance.csv') 
  #choose first 7 columns as features
  features = dataset.iloc[:,0:6] 
  #choose the final column for prediction
  labels = dataset.iloc[:,-1] 

  #one-hot encoding for categorical variables
  features = pd.get_dummies(features) 
  #split the data into training and test data
  features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
  #normalize the numeric columns using ColumnTransformer
  ct = ColumnTransformer([('normalize', Normalizer(), ['age', 'bmi', 'children'])], remainder='passthrough')
  #fit the normalizer to the training data and convert from numpy arrays to pandas frame
  features_train_norm = ct.fit_transform(features_train) 
  #applied the trained normalizer on the test data and convert from numpy arrays to pandas frame
  features_test_norm = ct.transform(features_test)

  #ColumnTransformer returns numpy arrays. Convert the features to dataframes
  features_train_norm = pd.DataFrame(features_train_norm, columns = features_train.columns)
  features_test_norm = pd.DataFrame(features_test_norm, columns = features_test.columns)
  # -----> your code here below
  # step 1: standardization
  my_ct = ColumnTransformer([('scale', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')

  # step 2: 
  """  fit the column transformer to the features_train DataFrame and at the same time transform it."""
  features_train_scale = my_ct.fit_transform(features_train)
  # step 3: transform the trained column transformer my_ct to the features_test DataFrame
  """To transform any pandas DataFrame using a ColumnTransformer instance ct"""
  features_test_scale = my_ct.transform(features_test)

  # step 4: Transform the features_train_scale 
  """NumPy"""# array back to a DataFrame using pd.DataFrame() assign the result back to a variable called features_train_scale .For the columns attribute use the .columns property of features_train.
  features_train_scale = pd.DataFrame(features_train_scale, columns = features_train.columns)
  # step 5: Transform the features_test_scale 
  """NumPy"""# array back to DataFrame using pd.DataFrame() and assign the result back to a variable called features_test_scale. For the 
  """columns""" # attribute use the .columns property of features_test
  features_test_scale = pd.DataFrame(features_test_scale, columns = features_test.columns)
  # step 6: print the statistics summary of the resulting train and test DataFrames, features_train_scale and features_test_scale
  """Observe the statistics of the numeric columns (mean, variance) use method .describe()"""
  print()
  print("the statistics of the numeric columns features_train_scale")
  print(features_train_scale.describe())
  print()
  print("the statistics of the numeric columns features_test_scale")
  print(features_test_scale.describe())

def neuralNetworkmodel():

  """Now that we have our data preprocessed we can start building the neural network model.
  The most frequently used model in TensorFlow in Keras Sequential . A sequential model, as the name suggests, allows us to create models layer-by-layer in a step-by-step fashion.
  This model can have only one input tensor and only one output tensor. """
  # To design a sequential model, 
  # first 1:  we first need to 
  """import Sequential """#from 
  """keras.models"""
  """
  To improve readability, we will design the model in a separate Python function
  called design_model(). The following command initializes a Sequential model instance
  my_model
  """
  # my_model = Sequential(name="my first model")
  """
  name is an optional argument to any model in 
  """
  # Keras.
  """Finally, we invoke our function in the main program with"""
  # my_model = design_model(features_train)
  """The model's layers are accessed via the """# layers attribute:
  # print(my_model.layers)
  """
  As expected, the list of layers is empty. In the next exercise, we will start adding layers to our model.  
  """
  print()


def design_model(features):

  # """name is an optional argument to
  # any model in tensorflow.keras.models.
  """sequential model, as the name suggests, allows us to create
  models layers-by-layer in a step-by-step fashion."""

  model = Sequential(name='my first model')
  return model
  








if __name__ == '__main__':
    # to upgrade pip for python 3: 
    # c:\users\abdo\appdata\local\programs\python\python39\python.exe -m pip install --upgrade pip
    # path for the lib machine learning C:\Users\abdo\AppData\Roaming\Python\Python39\site-packages
    print("...")
    print('practice about the method iloc[]')
    print('data.iloc[<row selection>, <column selection>]')
    print('')
    print("DEEP LEARNING")
    #load the dataset
    dataset = pd.read_csv('insurance.csv')
    #choose first row as features
    firstRow = dataset.iloc[0]
    # i will print the data now let's goooo
    print(firstRow)
    print()
    print("second row")
    print()
    secondRow = dataset.iloc[1]
    print(secondRow)
    # first column of the dataset
    firstColumn = dataset.iloc[:,0]
    print()
    print('first column of the dataset:vector index (,0)')
    print()
    print(firstColumn)
    # third column of the dataset
    thirdColumn = dataset.iloc[:,2]
    print()
    print('third column of the dataset:vector index (,2)')
    print()
    print(thirdColumn)
    # first row and first column of the dataset:vector index (0,0)
    firstRowFirstColumn = dataset.iloc[0,0]
    print()
    print('first row and first column of the dataset:vector index (0,0)')
    print()
    print(firstRowFirstColumn)
    # first row and second column of the dataset:vector index (0,0)
    firstRowSecondColumn = dataset.iloc[0,1]
    print()
    print('first row and first column of the dataset:vector index (0,0)')
    print()
    print(firstRowSecondColumn)   
    print()
    print(firstRowFirstColumn)
    # Second row and second column of the dataset:vector index (0,0)
    secondRowSecondColumn = dataset.iloc[2,2]
    print()
    print("second row and second column of the dataset:vector index (0,0) \n don't forget to know syntax  for iloc[<row selection>, <column selection>]")
    print()
    print(secondRowSecondColumn)
    # last column of the dataset:vector index (,n-1)
    lastColumn = dataset.iloc[:,-1]
    print()
    print("last column of the dataset:vector index (,n-1) \n don't forget to know syntax  for iloc[<row selection>, <column selection>]")
    print()
    print(lastColumn)
    # the second  column from the last  of the dataset:vector index (,n-1)
    secondColumnFromLast = dataset.iloc[:,-2]
    print()
    print("the second  column from the last  of the dataset:vector index (,n-2) \n don't forget to know syntax  for iloc[<row selection>, <column selection>]")
    print()
    print(secondColumnFromLast)
    # the third  column from the last  of the dataset:vector index (,n-1)
    thirdColumnFromLast = dataset.iloc[:,-3]
    print()
    print("the second  column from the last  of the dataset:vector index (,n-2) \n don't forget to know syntax  for iloc[<row selection>, <column selection>]")
    print()
    print(thirdColumnFromLast)
    print()
    """notice that the index of the column from the last is start by last column by iloc[:,-1] 
    the index for the second column from the last is iloc[:,-2] """

    print("notice that the index of the column from the last is start by last column by iloc[:,-1] \n the index for the second column from the last is iloc[:,-2] ")
    print()
    # choose the last three columns from dataset 
    chooseLastThreeColumns = dataset.iloc[:0,2]
    print()
    print("choose the last three columns from dataset ")
    print()
    print(chooseLastThreeColumns)
    # the property .shape return the couple (row, col) of the data
    # row =  dataset.shape[0]
    # col =  dataset.shape[1]
    # example
    print()
    print("To see the statistic of the data (like vector or two columns ) we use the method .describe() ")
    print()
    print(thirdColumnFromLast.describe())
    print()
    # choose first 7 columns as features
    features = dataset.iloc[:,0:6]
    # choose the final column for prediction
    labels = dataset.iloc[:,-1]

    # print the number of features in the dataset
    print("Number of features:", features.shape[1])
    # print the number of samples in the dataset
    print("Number of samples:", features.shape[0])
    # see useful summary statistics for numeric features
    print(features.describe())
    # your code here below
    print(labels.shape)
    print("------------ \n analyse of the data \n")
    print(labels.describe())
    # Since neural networks cannot work with string data directly
    # we need to convert our categorical features ("region") into numerical.
    print("convert the data")
    features  = pd.get_dummies(features)
    print()
    print("####################")
    print()
    print("we make the value of each kid of region either 1 or 0 \n all about the original value for the first data (table)")
    print()
    print(features)
    """tap this command pip install -U scikit-learn to install the lib sklearn"""
    
    """Here we chose the test size to be 33% of the total data,
    and random state controls the shuffling applied to be the
    data before applying the split."""
    print()
    print("Standardize/normalize numerical features:")
    print()
    """
    our features have different scales or units:
     “age” has an interval of [18, 64] and the “children” 
     column’s interval is much smaller, [0, 5].
    """
    # you can try them all out and choose the
    #  one method that gives the best results.
    # By having features with differing scales,
    #  the optimizer might update some weights faster than the others.
    print()
    print("call the function neural_network()")
    neural_network()
    print()
    dataset = pd.read_csv('insurance.csv')
    # load the dataset
    features = dataset.iloc[:,0:6] # choose first 7 comumns as features
    labels = dataset.iloc[:,-1]    # choose the final column for prediction

    features = pd.get_dummies(features)
    # one-hot encoding for categorical variables

    # split the data into training and test data
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

    """standardize"""
    ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')

    features_train = ct.fit_transform(features_train)
    features_test = ct.transform(features_test)

    # invoke the function for our model design
    model = design_model(features_train)

    # print the layers
    print(model.layers)
    # install python virtual environment , tap the command below
    #  c:\users\abdo\appdata\local\programs\python\python39\python.exe -m venv ./venv 
""""""
  








































































