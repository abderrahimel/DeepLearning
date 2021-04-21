# -*- coding: utf-8 -*-
"""
Welcome to the Build Deep Learning Models with TensorFlow Skill Path

"""
# Diving into deep learning! (apprentissage en profondeur)
"""
Welcome to the Build Deep Learning Models with TensorFlow skill path! Let's go over
what you will find along this path.
There are four main units:
. Foundations of Deep Learning and Perceptions
. Getting Started with TensorFlow
. Classification
. Deep Learning in the Real World

In each of these sections , you will dive into the concepts of deep learning and be 
prepared to put your skill to the test with quizzes and projects.

After this Path, you will be able to:
. identify use case for deep learning models
. implement a perceptron algorithm in python 
. preprocess data for various deep learning use cases
. build, train, and test deep learning models using TensorFlow
. tune hyperparameters to improve your models
. use regression models to draw predictions about data
. classify tabular and image data using deep learning models
. understand real-world applications of deep learning

You will demonstrate your knowledge in a  portfolio project at the end of the path
You can complete the Portfolio Project either in parallel with or after taking the prerequisite 
content --it's up to you!
Throughout this path you may see some items that aren't made by Codecademy team.
We're included these to rnsure that you learn all the topics you need to get your 
career started.Though we didn't write them, we've vetted them ourselves for technical
accuracy and good teaching practices.When possible, we're written relevant
assessments so that you can check your understanding back on our platform.
We're excited for you to begin this journey!

"""
# Additional Resources for Build Deep Learning Models
#  with TensorFlow Skill Path
"""
Book: Deep Learning with Python, François Chollet ==> https://bookshop.org/books/deep-learning-with-python/9781617294433
Documentation: Keras Library ==> https://keras.io/api/
Documentation: Tensorflow Library ==> https://www.tensorflow.org/overview/
Book: Algorithms of Oppression: How Search Engines Reinforce Racism, Safiya Umoja Noble ==> https://bookshop.org/books/algorithms-of-oppression-how-search-engines-reinforce-racism/9781479837243
Book: Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy, Cathy O’Neil
https://bookshop.org/books/weapons-of-math-destruction-how-big-data-increases-inequality-and-threatens-democracy/9780553418835
"""
# Introduction: Foundations of Deep Learning and Perceptrons
# The goal of this unit is to grasp the foundational concepts of deep learning (DL).
# Through articles and interactive applets, you will gain an insight 
# into the fundamentals of DL and move forward with knowledge of a powerful 
# and widely used machine learning method. You will then investigate perceptrons
#  as a prelude to the development of deep learning models to 
# get yourself acquainted with the basic structure of a neural network in Python.
"""
After this unit, you will be able to:

Explain what deep learning is to a friend
Know different use cases for deep learning models
Trace the beginning-to-end path of data that journeys a neural network and understand the inner workings behind the journey
Decipher potential dangers for the use cases that involve deep learning models
Develop your own perceptron algorithms
"""
# A quick overview of deep learning and its applications
 # Deep Learning vs. Machine Learning
"""Deep learning is a subfield of machine learning, and the concept 
of learning is pretty much the same.
"""

"""becoming more prevalent in today’s society.

  The deep part of deep learning refers to the numerous “layers” 
  that transform data. This architecture mimics the structure of the 
  brain, where each successive layer attempts to learn progressively complex patterns from the data fed into the model.
   This may seem a bit abstract, so let’s look at a concrete example, such as facial recognition. With facial recognition,
    a deep learning model takes in a photo as an input,
     and numerous layers perform specific steps to identify 
     whose face is in the picture. The steps taken by each layer might be the following:

  Find the face within the image using edge detection.
  Analyze the facial features (eyes, nose, mouth, etc.).
  Compare against faces within a repository.
  Output a prediction!"""
"""
This structure of many abstract layers makes deep learning incredibly 
powerful. Feeding high volumes of data into the model makes the connections between layers more intricate.
 Deep learning models tend to perform better with more massive amounts of data than other learning algorithms.
"""
#=====================================================================
# What are Neural Networks?

""" An artificial neural network is an interconnected group of nodes, an attempt to mimic to the vast network of neurons in a brain."""
# As you are reading this article, the very same brain 
# that sometimes forgets why you walked into a room is magically
#  translating these pixels into letters, words, and sentences —
#  a feat that puts the world’s fastest supercomputers to shame.
#  Within the brain, thousands of neurons are firing at incredible
#  speed and accuracy to help us recognize text, images,
#  and the world at large.
"""In 1957, Frank Rosenblatt explored the second question and invented the Perceptron algorithm that allowed an artificial neuron to simulate a biological neuron! The artificial neuron could take in an input, process it based on some rules, and fire a result. But computers had been doing this for years — what was so remarkable?

"""
# There was a final step in the Perceptron algorithm that would give rise to the incredibly mysterious world of Neural Networks — the artificial neuron could train itself based on its own results, and fire better results in the future. In other words, it could learn by trial and error, just like a biological neuron.
"""DEEP LEARNING MATH
Scalars, Vectors, and Matrices
To start, let us go over a couple of topics that will be integral 
to understanding the mathematical operations that are present 
in deep learning, including how data is represented:"""
# Scalars: A scalar is a single quantity that you can think of as a number.
# In machine learning models, we can use scalar quantities to manipulate data,
# and we often modify them to improve our model's accuracy .
# We can also represent data as scalar values depending on what dataset 
# we are working with.
# code example x = 4
"""vectors are arrays of numbers.In Python, we often denote vectors as NumPy arrays."""
# NumPy arrays code example x = np.array([1, 2, 3])
"""Matrice """
# code example x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
"""
Tensors
Scalars, vectors, and matrices are foundational objects in linear algebre. 
Understanding the different ways they interact with each other and can be manipulated through
matrix algebra is integral before diving into deep learning .
This is because the data structure we use in deep learning is called a tensor
,which is a generalized form of a vector and matrix; a multidimensional array.
  A tensor allows for mare flexibility with the type of data you are using and how can manipulate that data
"""
# This is because the data structure we use in deep learning is called a tensor
# Matrix Addition A + B
# Scalar Multiplication a * A
# Transpose A * B
"""Hidden layers"""#are layers that come between the input layer and the output layer.
"""You can have as many hidden layers as you want in a neural network (including zero of them )"""
"""The output layer is the final layer in our neural network. it produces the final result, so every neural network must have only one output layer."""
  # Each layer in a neural contains nodes.Nodes between each layer are connected by weights.
  # These are the learning parameters of our neural network, determining the strength of
  # of the connection between each linked node .
"""
The weighted sum between nodes and weights is calculated between each layer.
For example, from our input layer, we take the weighted sum of the inputs and our weights with the following equation:
weighted_sum = ( inputs * weight_transpose) + bias
Bias Node ===> is a node that 
 ===> 
       We then apply an activation function to it 

        Activation(weighted_sum)
"""
# we chose not to alter the slope of that function, never mind adding a constant. 
# We instead chose to change the weights of the incoming signals.
"""Bias Node"""
# The way to do this is to add a special additional node into a layer, alongside the others, which always has a constant value usually set to 1. The weight of the link is able to change, and even become negative. This has the same effect of adding the additional degree of freedom that we needed above.    
# 
"""
Let's bring all of these concepts together and see how they function in a neural network
with one hidden layer.
As you scroll over each section, you will see the inputs/weights/calculations associations
with it and see how inputs get from the starting point and make their way to the end!
The process we have been going through is known as forward propagation . Inputs are moved forward 
from the input layer through the hidden layer(s) unit they reach the output layer.

"""
"""
In this applet, you can scroll over each part of the neural network and observe the mathematics
the diagram.

When you scroll over the input section, you should see how the input is represented as a vector
Scrolling over the weights, we see how each set of weights (blue and yellow) is represented
 as a vector. When brought together, they make up the weights_matix and weights_matrix_transpose.
When scrolling through the hidden nodes sections, you will notice that there are two parts.
In the first step, we take the weighted sum of our data using the weights_matrix_transpose.
From this, we end up with a vector and apply our ReLU activation function to it.

This takes us to our teal weights.these are represented as a vector. The weights_teal_transpose turns our row
vector into a column vector.Then we take another weighted sum in output layer, this time between 
our hidden_nodes and our weights_teal_transpose Following this, we have a sigmoid activation function,
which gives us our output
Feel free to open this Applet in a separate window for a larger viewing screen
We now understand the adventure our data takes on one journey through our neural networkWe are not quite 
finished yet,thought. Let's keep exploring!

Input                Hidden Layer                         Out Layer
                     ____________                         ____________
                     |           |                        |           |--->  Output
                     |___________|                        |___________|
                     Hidden node 1  
                      
                     
Input
node 1
 __________ 
|           |                     ____________ 
|___________|                     |           |  
                                  |___________|  

input                               Hidden node 2
node 2
 ____________ 
 |           |
 |___________|

 input 
 node 3
 ____________
  |           |
  |___________|
https://content.codecademy.com/courses/deeplearning-with-tensorflow/deep-learning-math/applet_2_new.html

"""
# Loss Functions ( calculate error using a loss function)
# We have seen how we get to an output! Now, what do we do with it? When a value is outputted, we calculate its 
# error using a loss function
# Our predicted values are compared with the actual values within the training data:
# There are two commonly used loss calculation formulas:
# .Mean squared error, which is most likely familiar to you if you have come across linear regression.
# This gif below shows how mean squared error is calculated for a line of best fit in linear regression
# https://content.codecademy.com/courses/deeplearning-with-tensorflow/deep-learning-math/Loss.gif

# ** Cross-entropy loss, which is used for classification learning models rather than regression.
# You will learn more about this as you use loss functions in your deep learning models.

# check the link https://content.codecademy.com/programs/data-science-path/line-fitter/line-fitter.html
"""
the interactive visualization in the browser lets you try to find the line of best fit for 
a random set of data points:
we want this loss to be as small as possible.
To check if you got the best line, check the "Plot Best-Fit" box
"""
# Play around with the interactive applet, and notice what method you use to minimize loss:
# Do you first get the slope to where it produces lowest loss, and then move the intercept to where it produces lowest loss?
# Do you create a rough idea in your mind where the line should be first, and then enter the parameters to match that image?
"""gradient descent.This algorithm continuously updates and refines the weights between neurons
to minimize our loss function.
"""
# deep learning: This is where backpropagation and gradient descent come into play.

"""
Forward propagation deals with feeding the input values through hidden layers to the final output layer. 
"""
# Backpropagation refers to the computation of gradients with an algorithm known as gradient descent.
# This algorithm continuously updates and refines the weights(poids: les valeur entre les nœuds) between neurons to minimize our loss function
"""
  Backpropagation 
    there is more to our deep learning models.
This is where backpropagation and gradient descent come into play.Forward propagation
deals with feeding the inpute values through hidden layers to the final output layer.
Backpropagation refers to the computation of gradients with an algorithm knowns as gradient descent.
This algorithm continuously updates and refines the weights between neurons to minimize our loos function.

By gradient, we mean the rate of change with respect to the parameters of our loss function.From this,
backpropagation determines how much each weight is contributing to the error in our loss function,
and gradient descent will update our weight values accordingly to decrease this error.
  This is a conceptual overview of backpropagation.
  if you would like to engage with the gritty mathematics of it, you
  can do so "https://en.wikipedia.org/wiki/Backpropagation" .
"""
# Let's take a look at what happens with backpropagation and pradient descent on a neural network directly
# In the applet in the learning environement "https://content.codecademy.com/courses/deeplearning-with-tensorflow/deep-learning-math/interactives/index.html", watch as weights are updated and error is decreased after each iteration.
#Without backpropagation, neural network would be much less accurate
"""gradient discent"""
# Because of this, performing backpropagation 
# and gradient descent calculations on all of our data may 
# be inefficient and computationally exhaustive no matter
#  what learning rate we choose.
"""To solve this problem, a variation of gradient descent
known as Stochastic Gradient Descent (SGD) was developed."""
# This is where SGD comes to play. Instead of performing 
# gradient descent on our entire dataset, we pick out a 
# random data point to use at each iteration. 
# This cuts back on computation time immensely 
# while still yielding accurate results.
""""
Stochastic Gradient Descent.
Gradient Descent
"""
# linear function y = a * x + p 
# w0 + w1 * x and w0, w1 ... are  weights of the neural network

"""
In the field of artificial neural networks, the activation function is a mathematical function applied to a signal
 at the output of an artificial neuron. The term “activation function” comes from the biological equivalent of “activation potential”,
  a stimulation threshold which, once reached, results in a response from the neuron. The activation function is often a non-linear function.
   An example of an activation function is the Heaviside function, which always returns 1 if the input signal is positive, or 0 if it is negative.
"""
################################################################
# DEEP LEARNING MATH
"""Review"""
"""
This overview completes the necessary mathematical intuition you need to 
move forward and design coding your own learning models! To recap all the things we have learned (so many things)

. Scalars, vectors, matrices, and tensors
  
      .  A scalar is a singlar quantity like a number.
      .  A vector is an array of numbers (scalar values).
      .  A matrix is a grid of information with rows and columns.
      A tensor is a multidimensional array and is a generalized version of a vector and matrix.
. Matrix Algebra
    . In scalar multiplication, every entry of the matrix is multiplied by a scalar value.
    . In matrix multiplication, the dot production between the corresponding rows of the first matrix and columns of the second matrix is calculated.
    . A matrix transpose turns the rows of a matrix into columns.
. In forward propagation, data is sent through a neural network to get initial outputs and error values
"""
"""
. Weights are the learning parameters of a deep learning model that determine the strength of the connection between two nodes.
. A bias node shifts the activation function either left or right to create the best fit for the given data in a deep learning model.
. Activation Functions are used in each layer of a neural network and determine whether neurons should be “fired” or not based on output from a weighted sum.
. Loss functions are used to calculate the error between the predicted values and actual values of our training set in a learning model.

  ===> In backpropagation, the gradient of the loss function is calculated with respect to the weight parameters within a neural network.

. Gradient descent updates our weight parameters by iteratively minimizing our loss function to increase our model’s accuracy.
. Stochastic gradient descent is a variant of gradient descent, where instead of using all data points to update parameters, a random data point is selected.
. Adam optimization is a variant of SGD that allows for adaptive learning rates.
. Mini-batch gradient descent is a variant of GD that uses random batches of data to update parameters instead of a random datapoint.


"""
# Representing Perceptron

"""
so the perceptron is an artificial neuron that can make a simple decision .Let's implement one from scratch
 in Python
The perceptron has three main components:

Inputs : Each input corresponds to a feature .For example, in the case of a person, feature could be,
 height, weight, college degree.

Weights: Each input also has a weight which assigns a certain amount of importance to the input.
 if an input's weight is large, it means this input plays a bigger role in determining the output For example,
  a team’s 
skill level will have a bigger weight than the average age of players in determining the outcome of a match.
Output: Finally, the perceptron uses the inputs and weights to produce an output.
 The type of the output varies depending on the nature of the problem. For example, to predict whether
  or not it’s going to rain, the output has to be binary — 1 for Yes and 0 for No. However, to predict 
the temperature for the next day, the range of the output has to be larger — say a number from 70 to 90.


"""
# PERCEPTRON

# Step 1: Weighted Sum
"""the first step is finding the weighted sum of the inputs."""
"""
What is the weighted sum? This is just a number that gives a reasonable representation of the inputs:

weighted sum = x_1 * w_1 + x_2 * w_2 + ... + x_n * w_n
w1, w2  .... are weights (poids in the link)
"""
"""Here’s how we can implement it:"""
#Start with a weighted sum of 0. Let’s call it weighted_sum.
#Start with the first input and multiply it by its corresponding weight. Add this result to weighted_sum.
#Go to the next input and multiply it by its corresponding weight. Add this result to weighted_sum.
#Repeat this process for all inputs.

"""Step 2: Activation Function"""
#After finding the weighted sum, the second step is to constrain 
# the weighted sum to produce a desired output.

# How can the perceptron produce a meaningful output in this case? 
# This is exactly where activation functions come in! 
# These are special functions that transform the weighted sum into 
# a desired and constrained output.
# For example, if you want to train a perceptron to detect whether
#  a point is above or below a line (which we will be doing in this lesson!),
#  you might want the output to be a +1 or -1 label. For this task, 
# you can use the “sign activation function” to help the perceptron 
# make the decision:

# if weighted sum is positive, return +1
# if weighted sum is negative, return -1
# in this lesson , we will focus on using 
# the sign activation function because it is the simplest
#  way to get started with perceptrons and eventually visualize one in action .

"""Training the Perceptron"""

#Our perceptron can now make a prediction given inputs,
#  but how do we know if it gets those predictions right?

# Right now we expect the perceptron to be very bad
#  because it has random weights. We haven’t taught it
#  anything yet, so we can’t expect it to get classifications
#  correct! The good news is that we can train the
#  perceptron to produce better and better results! 
# In order to do this, we provide the perceptron a 
# training set — a collection of random inputs with 
# correctly predicted outputs.

# On the right, you can see a plot of scattered points 
# with positive and negative labels. This is a simple training set.

# In the code, the training set has been represented 
# as a dictionary with coordinates as keys and labels as values. For example:

"""training_set = {(18, 49): -1, (2, 17): 1, (24, 35): -1, (14, 26): 1, (17, 34): -1}
"""

# We can measure the perceptron’s actual performance against this training set.
#  By doing so, we get a sense of “how bad” the perceptron is.
#  The goal is to gradually nudge the perceptron — 
# by slightly changing its weights — towards a better version of itself
#  that correctly matches all the input-output pairs in the training set.

# We will use these points to train the perceptron to correctly separate 
# the positive labels from the negative labels by visualizing the perceptron
#  as a line. Stay tuned!

"""Training Error
"""
# Now that we have our training set, we can start feeding inputs 
# into the perceptron and comparing the actual outputs against 
# the expected labels!

# Every time the output mismatches the expected label, 
# we say that the perceptron has made a training error — 
# a quantity that measures “how bad” the perceptron is performing.
"""a training error — a quantity that measures “how bad” the perceptron is performing."""

"""As mentioned in the last exercise, the goal is to nudge the perceptron towards zero training error. """
# The training error is calculated by subtracting the predicted label value from the actual label value.
"""training error=actual label−predicted label
"""
#For each point in the training set, 
"""the perceptron either produces a +1 or a -1 
 (as we are using the Sign Activation Function).
  Since the labels are also a +1 or a -1, 
 there are four different possibilities for the error the perceptron makes:"""

#    Actual             Predicted              Training Error

#       +1                  +1                        0

#       +1                  -1                        2

#       -1                   -1                       0

#       -1                    +1                      -2


#These training error values will be crucial in improving the perceptron's
# performance as we will see in the upcoming exercises.

"""Tweaking the Weights
"""

# What do we do once we have the errors for the perceptron? 
# We slowly nudge the perceptron towards a better version of itself 
# that eventually has zero error.

# The only way to do that is to change the parameters that define the perceptron. 
# We can’t change the inputs so the only thing that can be tweaked are the weights. 
# As we change the weights, the outputs change as well.

"""
** The goal is to find the optimal combination of weights that will produce the correct output for as many points as possible in the dataset.

"""
#Tweaking the Weights
"""What do we do once we have the errors for the perceptron? We slowly nudge the perceptron towards a better version of itself that eventually has zero error.

The only way to do that is to change the parameters that define the perceptron. We can’t change the inputs so the only thing that can be tweaked are the weights. As we change the weights, the outputs change as well.

The goal is to find the optimal combination of weights that will produce the correct output for as many points as possible in the dataset.
"""
#               The Perceptron Algorithm
"""
But one question still remains --- how do we tweak weights optimally? We cn't just play 
around randomly with weights until the correct combination magically pops up. There needs to be a way to guarantee that the perceptron improves its performance over time.

This is where the Perceptron Algorithm comes in. The math behind why this works is outside scope of this lesson, so we'll directly apply the algorithm to optimally tweak the weights and nudge the perceptron towards zero error.

The most important part of the algorithm is the update rule where the weights get updated:

     weight = weight + (error * input)

We keep on tweaking the weights until all possible labels are correctly predicted by the perceptron. This means that multiple passes might need to be made through the training_set before the Perceptron Algorithm comes to a half.

In this exercise, you will continue to work on the .training( ) method. We have made the following changes to this method from the last exercise:

     . foundLine = False (boolean that indicates whether the perceptron has found a line to separate the positive and negative labels)
     . while not foundLine: ( a while loop that continues to train the perceptron until the line is found)
     . total_error = 0 (to count the total error the perceptron makes in each round)
     . total_error += abs(error)  (to update the total error the perceptron makes in each round)

"""
# The bias Weight

"""
The Bias Weight

you have understood that the perceptron can be trained to produce correct outputs by tweaking the regular weights.

However,there are times when a minor adjustement is needed for the perceptron to be more accurate.
This supporting role is played by the bias weight.
It takes a default input value of 1 and some random weight value.

So now the weighted sum equation should look like:

 weighted_sum = x1 * w1 + x2 * w2 + ... + xn * wn + 1Wb

How does this change the code so far? You only have to consider two small changes:

    .  Add a 1 to the set of inputs (now there are 3 inputs instead of 2)
    . Add a bias weight to the list of weights (now there are 3 weights instead of 2)

We'll automatically make these replacement in the code so you should be good to go!

"""
# Representing a Line 

"""
so far so good! The perceptron works as expected, but everything seems to be taking place behind the scenes .
What if we could visualize the perceptron's training process to gain a better undestanding of what's going on?

The weights change throughout the training process so if only we could meaningfully visualize those weights ...

Turns out we can! In fact, it gets better. The weights can actually be used to represent a line! This greatly simplifies our visualization.

You might know that a line can be represented using the slop-intercept form.
A perceptron's weights can be used to find the slope and intercept of the line that the perceptron represents.

  . slope = -self.weights[0]   /   self.weights[1]
  intercept = -self.weights[2] /   self.weights[1]

The explanation for these equations is beyond the scope of this lesson, so we'll just use them to visualize the perceptron for now.

In the plot on your right, you should be able to see a line that represents the perceptron in its first iteration of the training process.


"""
# Finding a Linear Classifier
"""
Let’s recap what you just learned!

The perceptron has inputs, weights, and an output. The weights are parameters that define the perceptron and they can be used to represent a line. In other words, the perceptron can be visualized as a line.

What does it mean for the perceptron to correctly classify every point in the training set?

Theoretically, it means that the perceptron predicted every label correctly.

Visually, it means that the perceptron found a linear classifier, or a decision boundary, that separates the two distinct set of points in the training set.

In the plot on the right, you should be able to see the linear classifier that was found by the perceptron in the last iteration of the training process.

"""

# What's Next? Neural Networks

"""

Congratulations! You have now built your own perceptron from scratch.

Let’s step back and think about what you just accomplished and see if there are any limits to a single perceptron.

Earlier, the data points in the training set were linearly separable i.e. a single line could easily separate the two dissimilar sets of points.

What would happen if the data points were scattered in such a way that a line could no longer classify the points? A single perceptron with only two inputs wouldn’t work for such a scenario because it cannot represent a non-linear decision boundary.

That’s when more perceptrons and features come into play!

By increasing the number of features and perceptrons, we can give rise to the Multilayer Perceptrons, also known as Neural Networks, which can solve much more complicated problems.

With a solid understanding of perceptrons, you are now ready to dive into the incredible world of Neural Networks!
"""
# quiz for perceptron 

"""1 -Suppose the actual label is 4 and the predicted label is -2 for a data point, what is the taining error?"""
# 6
"""2 - Which of the following is the correct weighted sum formula?"""
# weighted_sum = x1 * w1 + x2 * w2 + ... + xn * wn

"""3 - True/False: An activation function is a special function that transforms the weighted sum into a desired and constrained output."""
# True
"""4 - Which of the following is not a component of perceptron?"""
# inputs (x)
# output(y)
# weights (w)
"""5 - What was used to find slope and intercept of the line that the perceptron represents?"""
# A perceptron's weights are used to represent a line.
"""6 - What does the activation function below do?"""
"""
def activation(self, weighted_sum):
  if weighted_sum >= 0:
    return 1
  if weighted_sum < 0:
    return -1
"""
# if the weighed sum is positive, return 1 and if the sum is negative, return -1
"""7 - What is the default input value for the bias weight?"""
"""
weight= x1	* w1	 +x2	* w2​	 +...+xn * wn	 +?wb
​	
"""
# 1

# =========================================================================

"""FOUNDATIONS OF DEEP LEARNING AND PERCEPTRONS"""
# in this project, we will use perceptrons to model the fundamental building blocks of computers -- logic gates(les portes logic)
"""
Name                                   AND                            OR                     XOR

Symbol                              a ---
                                      |  |----- X                    SymbolOR                SymbolXOR
                                    b ---

Truth                               a   b |  X                        a   b |  X             a   b |  X     
Table                               0   0    0                        0   0    0             0   0    0
                                    1   0    0                        1   0    1             1   0    1
                                    0   1    0                        0   1    1             0   1    1
                                    1   1    1                        1   1    1             1   1    0
"""

# For example, the table above shows the results of an AND gate.
#  Given two inputs, an AND gate will output a 1 only if both inputs are a 1:

# We’ll discuss how an AND gate can be thought of as linearly separable data and
#  train a perceptron to perform AND.

# We’ll also investigate an XOR gate — a gate that outputs
#  a 1 only if one of the inputs is a 1:

"""We’ll think about why an XOR gate isn’t linearly separable and show 
how a perceptron fails to learn XOR.

"""
##   Perceptron Logic Gates    {AND, OR, XOR}

"""
import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
# Creating and visualizing AND Data
# list contains the four possible inputs to an AND
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# each item in the lables correspand to the output of the input
labels = [0,      0,       0,      1]
#plot these four points on a grash.
plt.scatter([point[0] for point in data],
             [point[1] for point in data],
               c=labels)
# Building the Perceptron

max_iter is the number of times the perceptron loops through the training data. 
the default is 1000, so we're cutting the training pretty short! Let's see if our algorithm learns AND even with very little traing 

classifier =  Perceptron(max_iter=40)
# We'll now train the model, Call the .fit() method using data and labels as parameters.
classifier.fit(data, labels)

Even though an input like [0,5, 0.5] isn't a real input to an AND logic gate, we can still check to see how far it is from the decision boundary.
We could also do this to the point [0, 0.1], [0, 0.2] and so on. if we do this for a grid of points, we can make a heat map that reveals the decision boundary.
To begin, we need to create a list of the points we want to input to .decision_function().
Begin by creating a list named x_values .
x_values should be a list of 100 evently spaced decimals between 0 and 1. 
np.linspace(0, 1, 100)
do the same for y_values.

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

We have a list of 100 x values x and values and 100 y values.
We now want to find every possible combination of those x and y values.
The function production will do this for you. For example, consider the following code:
list(product([1, 2, 3], [4, 5, 6]))

This code will produces the following list:
[(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6),(3, 4), (3, 5), (3, 6)]
Call product() using x_values and y_values as parameters. Don't forget to put list() around the call to product().Store the result in a variable named point_grid.

point_grid = list(product(x_values, y_values))

Call classifier 's .decision_function() method using point_grid as a parameter.Store the results in a variable named distances
distances = classifier.decision_function(point_grid)

Right now distances stores positive and negative values . We only care about how far away a point is from the boundary __ we don't care about the sign.
Take the absolute value of every distance .Use list comprehension to call abs() on every point in the list and store it in a new variable called abs_distances


abs_distances = [abs(pt) for pt in distances]

We're almost ready to draw the heat map we're going to be using Matplotlib's pcolormesh() function.
Right now, abs_distances is a list of 10000 numbers.pcolormesh needs a two dimensional list.we need to turn abs_distances into a 100 by 100 dimensional array.
Numpy's reshape function does this for us . The code below turns list lst into  a 2 by 2 list.
lst = [1, 2, 3, 4]
new_lst = np.reshape(lst, (2, 2))
new_lst now looks like this:
[[1, 2], [3, 4]]
Turn abs_distances_matrix


distances_matrix = np.reshape(abs_distances, (100, 100))

Let's see if the algorithm learned AND. Call classifier 's .score() method using data and labels as parameters.
This will print accuracy of the model on the data points.
* Note that it is pretty unusual to train and test on the same dataset. In this case, since there are only four possible inputs to AND, we're stuck training on every possible input and testing on those same points.

print(classifier.score(data, labels))
# Your perceptron should have 100% accuracy! You just taught it an AND gate!
# Let's change the labels so your data now represents an XOR gate.The label should be a represents an XOR gate. The label should be a 1 only if one of the inputs is a 1.
# What is the accuracy of the perceptron now? Is the data linearly separable? you can change to other Logic Gates

Visualizing the Perceptron 
We know the perceptron has been trained coorectly, but let's try to visualize what decision boundary it is making.Reset your labels to be representing an AND gate.
Let's first investigate the classifier's .decision_function() method.Given a list of points, this method returns the distance those points are from the decision boundary.The closer the number is to 0,the closer that point is to the decision boundary
Try calling classifier 's .decision_function() method using [[0, 0], [1, 1], [0.5, 0.5]] as a parameter.Print the results.
Is the point [0, 0] or the point [1, 1] closer to the decision boundary?


print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

it's finally time  to draw the heat map! Call 
plt.pcolormesh() with following three parameters:
     . x_values

     . y_values

     . distances_matrix
Save the result in a variable named heatmap.
  Then call plt.colorbar() using heatmap as a parameter. This will put a legend on the heat map.
  Make sure plt.show() is still below these function calls.

heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)

plt.colorbar(heatmap)

plt.show()

Great work! You now have a great visualization of what the perceptron is doing.You should see a purple line where the distances are 0. That's the decision boundary!

Change your labels back to representations an OR gate .Where does the decision boundary go?

Change your labels to represent an XOR gate .
Remember, this data is not linearly separable.
Where does the decision boundary go?

Perceptron can't solve problems that aren't linearly separable. However, if you combine multiple perceptrons together, you now have a neural net that can solve  these problems!

This is incredibly similar to logic gates.AND gates and OR gates can't produce the output of XOR gates, but when you combine a few ANDs and ORs, you can make an XOR!
"""

#####################################################################
"""
there is no way here that i can draw a line where all the black dots will be on one side and all the yellow dots will be on other side, so this data is not linearly
separable.

   . XOR gate 
-----------------------------------------------------------------------------------
   data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# each item in the lables correspand to the output of the input
labels = [0,      1,       1,      1]
#plot these four points on a grash.
plt.scatter([point[0] for point in data],[point[1] for point in data],c=labels)
-----------------------------------------------------------------------------------
   
                                               |          . (yelow pot)                      . (black pot) 
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |          . (black pot)                        . (yellow pot)
                                               |
                                               |_____________________________________________________

the function classifier.score()  returned 0.5, so the percent is not 100% ,so this data isn't linearly separable
===============================================================================================================
   . AND gate
   -----------------------------------------------------------------------------------
   data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# each item in the lables correspand to the output of the input
labels = [0,      0,       0,      1]
#plot these four points on a grash.
plt.scatter([point[0] for point in data],[point[1] for point in data],c=labels)
-----------------------------------------------------------------------------------

                                               |          . (black pot)                        . (yellow pot) 
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |          . (black pot)                        . (black pot)
                                               |
                                               |_____________________________________________________

the function classifier.score()  returned 1.0, so the percent is  100% ,so this data is linearly separable
===============================================================================================

   .OR gate

-----------------------------------------------------------------------------------
   data = [[0, 0], [0, 1], [1, 0], [1, 1]]
# each item in the lables correspand to the output of the input
labels = [0,      1,       1,      1]
#plot these four points on a grash.
plt.scatter([point[0] for point in data],[point[1] for point in data],c=labels)
-----------------------------------------------------------------------------------
                                               |          . (black pot)                      . (black pot) 
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |
                                               |          . (yellow pot)                        . (black pot)
                                               |
                                               |_____________________________________________________

the function classifier.score()  returned 1.0, so the percent is  100% ,so this data is linearly separable
                                          
"""
"""                                                    Review: Foundations of Deep Learning and Perceptrons
"""

"""
Foundations Deep Learning and Perceptrons

"""
# Goals of this Unit

"""
Congratulations! The goal of this unit was to grasp the foundational concepts of deep learning (DL).You gained an insight into the fundamentals of DL and investigated
investigated perceptrons as a preview to deep learning and to get acquainted with the basic structure of a neural network. You are ready to move forward with investigating deep 
learning models with Python!

Having completed this unit, you are now able to:
   .Explain what deep learning is to a friend
   . Known different use cases for deep learning models
  . trace the begining-to-end path of data that journeys a neural network and
  understand the inner workings behind the journey

  . Decipher potential for the use cases that involve deep learning models
  . Develop your own perceptron algorithms

Happy coding!
"""
##########################################################################################################################
##
"""
Congrats!
You just finished Review: Foundations of Deep Learning and Perceptrons"""


################################################################################################################"
# Introduction: Getting Started with TensorFlow
"""Create your own neural networks using TensorFlow!

"""
"""
Implementing Neural Networks Goals of this Unit

"""
#The goal of this unit is to take the abstract concepts about deep learning you have learned 
# and translate them into Python using TensorFlow.
# in this section, you will find code your own neural and investigate ways to fine-tune your program with
# hyperparameters. After exploring these concepts in lesson, you will get to work on you own regression projects where you will have to design and develop
# your ownneural network to solve a given problem.

#After this unit, you will be able to:
#    . Translate abstract deep learning concepts into Python
#    . Design a neural network from scratch
#    . Use deep learning models to solve regression problems
# Tune the hyperparameters of you model and improve its performance through
# Happy coding Let's goooooooooooooooooooooooooo
"""You just finished Introduction: Getting Started with TensorFlow

"""
### Implementing Neural Networks
"""Introduction
A neural network, just like any machine learning method,learns how to perform tasks by processing data
and adjusting its model to best predict the desired outcome.
Most popular machine learning tasks are:
"""
######
"""1. Classification"""
# . Classification: given data and true labels or categories for each data 
# point, train a model that predicts for each data example what its label should 
# be. For example, given data of previous fire hazards, our model can learn how to predict whether 
# a fire will occur for a given day in the future, with all the factors taken into account.
"""2 . Regression"""
#Regression:given data and true continuous value for each data point, train a model that can predict values
# the previous stock market data,we can build a regression model that forecasts what the stock
# market price will be at a specific point in time when the data is available.

"""
Parametric models such as neural networks are described by parameters: configuration variables representing the model's 
knowledge. We can tweak the parameters using the training data and we can evaluate the
the performance of the model using hold-out test data the model has not seen during
training
"""
#========================================================================================================
"""Take a look at the main components of a neural network learning pipeline depicted in the workspace:
"""
"""Input data"""
#  . Input data: this is used to train a neural network model you need to provide it with some training data
"""An optimizer"""
#An optimizer: this is an algorithm that based on the training data adjusts the parameters of the network in order to perform the task at hand.
"""A loss or cost function"""
# A loss or cost function: this informs the optimizer whether it is doing a good job on the training data and how to adjust the parameters in the right direction.
"""Evaluation metrics"""
# Evaluation metrics: these tell us how well the current model performs on validation data . For example, mean absolute error for regression tells us how far the predictions are on average from the true values.
"""
                                              ________________
                                              |               |                    
                                              | error/loss    |
                                 ------------>| cost function |<--------------------- |
                                |             |               |                       |
                                |             |_______________|                       |
                                |                   |                 |               |
                                |                   |                 |               |       
                                |                   |                 |               |                      
                                |                   | to buttom       | backward pass |      
                                |                   |                 |               |
                                |                   |                 |               | predicted
                                | true         _______________        |               |  outputs
                                |              |              |       |               |
                                | labels       |  optimizer   |       | to bottom     |
                                |              |              |       |               |
                                |              |______________|       |               |
                                |                   |                 |               |
                                |                   |                 |               |
                                |                   |   to buttom     |               |
                                |                   |                 |               |
                                |                   |                 |               |
                                |         __________|___________      |     __________|___________
                      __________|______   |        model        |          |       output         |      
                      |               |   |  neurol network     |--------->|   predictions from   | 
                      |               |-->|                     |          |                      | 
                      | input data    |   |_____________________|          |______________________| 
                      |features+labels|      
                      |_______________|                         
                                ---------------------------------------------->  
                                          forward pass               
program EEG                                                            
"""

"""
#############################"""








##################################################################
class Perceptron:
  def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
  
  def training(self, training_set):
    """The Perceptron Algorithm"""
    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
      if total_error == 0:
        foundLine = True
      


if __name__ == '__main__':

  print("")

  print(" Step 1: Weighted Sum")
  cool_perceptron = Perceptron()
  print(cool_perceptron.weighted_sum([24, 55]))

  #  la fonction d'activation est une fonction mathématique appliquée à un signal en sortie d'un neurone artificiel
  print("Step 2: Activation Function :")
  print("the result of the method activation is {result}".format(result=cool_perceptron.activation(52)))
  print("awesome let's gooo")
  print("The Perceptron Algorithm")
  small_training_set = {(0,3):1, (3,0):-1, (0,-3):-1, (-3,0):1}
  cool_perceptron.training(small_training_set)
  print(cool_perceptron.weights)





