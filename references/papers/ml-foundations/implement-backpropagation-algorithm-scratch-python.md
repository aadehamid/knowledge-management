title: How to Code a Neural Network with Backpropagation In Python \(from scratch\) - MachineLearningMastery.com How to Code a Neural Network with Backpropagation In Python \(from scratch\) - MachineLearningMastery.com
description: The backpropagation algorithm is used in the classical feed-forward artificial neural network. It is the technique still used to train large deep learning networks. In this tutorial, you will discover how to implement the backpropagation algorithm for a neural network from scratch with Python. After completing this tutorial, you will know: How to forward-propagate an \[…\] The backpropagation algorithm is used in the classical feed-forward artificial neural network. It is the technique still used to train large deep learning networks. In this tutorial, you will discover how to implement the backpropagation algorithm for a neural network from scratch with Python. After completing this tutorial, you will know: How to forward-propagate an input to calculate an output.…
author: Jason Brownlee

### [Navigation](#navigation)

# How to Code a Neural Network with Backpropagation In Python (from scratch)

By [Jason Brownlee](https://machinelearningmastery.com/author/jasonb/) on October 22, 2021 in [Code Algorithms From Scratch](https://machinelearningmastery.com/category/algorithms-from-scratch/ "View all items in Code Algorithms From Scratch") [ 845](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comments)

The backpropagation algorithm is used in the classical feed-forward artificial neural network.

It is the technique still used to train large [deep learning](https://machinelearningmastery.com/what-is-deep-learning/) networks.

In this tutorial, you will discover how to implement the backpropagation algorithm for a neural network from scratch with Python.

After completing this tutorial, you will know:

- How to forward-propagate an input to calculate an output.
- How to back-propagate error and train a network.
- How to apply the backpropagation algorithm to a real-world predictive modeling problem.

**Kick-start your project** with my new book [Machine Learning Algorithms From Scratch](https://machinelearningmastery.com/machine-learning-algorithms-from-scratch/), including *step-by-step tutorials* and the *Python source code* files for all examples.

Let’s get started.

- **Update Nov/2016**: Fixed a bug in the activate() function. Thanks Alex!
- **Update Jan/2017**: Fixes issues with Python 3.
- **Update Jan/2017**: Updated small bug in update\_weights(). Thanks Tomasz!
- **Update Apr/2018**: Added direct link to CSV dataset.
- **Update Aug/2018**: Tested and updated to work with Python 3.6.
- **Update Sep/2019**: Updated [wheat-seeds.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv) to fix formatting issues.
- **Update Oct/2021**: Reverse the sign of error to be consistent with other literature.

![How to Implement the Backpropagation Algorithm From Scratch In Python](https://machinelearningmastery.com/wp-content/uploads/2016/11/How-to-Implement-the-Backpropagation-Algorithm-From-Scratch-In-Python.jpg){width=640 height=512}

How to Implement the Backpropagation Algorithm From Scratch In Python\
Photo by [NICHD](https://www.flickr.com/photos/nichd/21086425615/), some rights reserved.

## Description

This section provides a brief introduction to the Backpropagation Algorithm and the Wheat Seeds dataset that we will be using in this tutorial.

### Backpropagation Algorithm

The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks.

Feed-forward neural networks are inspired by the information processing of one or more neural cells, called a neuron. A neuron accepts input signals via its dendrites, which pass the electrical signal down to the cell body. The axon carries the signal out to synapses, which are the connections of a cell’s axon to other cell’s dendrites.

The principle of the backpropagation approach is to model a given function by modifying internal weightings of input signals to produce an expected output signal. The system is trained using a supervised learning method, where the error between the system’s output and a known expected output is presented to the system and used to modify its internal state.

Technically, the backpropagation algorithm is a method for training the weights in a multilayer feed-forward neural network. As such, it requires a network structure to be defined of one or more layers where one layer is fully connected to the next layer. A standard network structure is one input layer, one hidden layer, and one output layer.

Backpropagation can be used for both classification and regression problems, but we will focus on classification in this tutorial.

In classification problems, best results are achieved when the network has one neuron in the output layer for each class value. For example, a 2-class or binary classification problem with the class values of A and B. These expected outputs would have to be transformed into binary vectors with one column for each class value. Such as \[1, 0\] and \[0, 1\] for A and B respectively. This is called a one hot encoding.

### Wheat Seeds Dataset

The seeds dataset involves the prediction of species given measurements seeds from different varieties of wheat.

There are 201 records and 7 numerical input variables. It is a classification problem with 3 output classes. The scale for each numeric input value vary, so some data normalization may be required for use with algorithms that weight inputs like the backpropagation algorithm.

Below is a sample of the first 5 rows of the dataset.

1

2

3

4

5

15.26,14.84,0.871,5.763,3.312,2.221,5.22,1

14.88,14.57,0.8811,5.554,3.333,1.018,4.956,1

14.29,14.09,0.905,5.291,3.337,2.699,4.825,1

13.84,13.94,0.8955,5.324,3.379,2.259,4.805,1

16.14,14.99,0.9034,5.658,3.562,1.355,5.175,1

Using the Zero Rule algorithm that predicts the most common class value, the baseline accuracy for the problem is 28.095%.

You can learn more and download the seeds dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/seeds).

Download the seeds dataset and place it into your current working directory with the filename **seeds\_dataset.csv**.

The dataset is in tab-separated format, so you must convert it to CSV using a text editor or a spreadsheet program.

Update, download the dataset in CSV format directly:

- [Download Wheat Seeds Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)

## Tutorial

This tutorial is broken down into 6 parts:

1. Initialize Network.
2. Forward Propagate.
3. Back Propagate Error.
4. Train Network.
5. Predict.
6. Seeds Dataset Case Study.

These steps will provide the foundation that you need to implement the backpropagation algorithm from scratch and apply it to your own predictive modeling problems.

### 1. Initialize Network

Let’s start with something easy, the creation of a new network ready for training.

Each neuron has a set of weights that need to be maintained. One weight for each input connection and an additional weight for the bias. We will need to store additional properties for a neuron during training, therefore we will use a dictionary to represent each neuron and store properties by names such as ‘**weights**‘ for the weights.

A network is organized into layers. The input layer is really just a row from our training dataset. The first real layer is the hidden layer. This is followed by the output layer that has one neuron for each class value.

We will organize layers as arrays of dictionaries and treat the whole network as an array of layers.

It is good practice to initialize the network weights to small random numbers. In this case, will we use random numbers in the range of 0 to 1.

Below is a function named **initialize\_network()** that creates a new neural network ready for training. It accepts three parameters, the number of inputs, the number of neurons to have in the hidden layer and the number of outputs.

You can see that for the hidden layer we create **n\_hidden** neurons and each neuron in the hidden layer has **n\_inputs \+ 1** weights, one for each input column in a dataset and an additional one for the bias.

You can also see that the output layer that connects to the hidden layer has **n\_outputs** neurons, each with **n\_hidden \+ 1** weights. This means that each neuron in the output layer connects to (has a weight for) each neuron in the hidden layer.

1

2

3

4

5

6

7

8

\# Initialize a network

def initialize\_network ( n\_inputs ,  n\_hidden ,  n\_outputs ) :

network  \=  list ( )

hidden\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_inputs  \+  1 ) \] }  for  i  in  range ( n\_hidden ) \]

network . append ( hidden\_layer )

output\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_hidden  \+  1 ) \] }  for  i  in  range ( n\_outputs ) \]

network . append ( output\_layer )

 return network

Let’s test out this function. Below is a complete example that creates a small network.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

from random import seedfrom random import random # Initialize a network

def initialize\_network ( n\_inputs ,  n\_hidden ,  n\_outputs ) :

network  \=  list ( )

hidden\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_inputs  \+  1 ) \] }  for  i  in  range ( n\_hidden ) \]

network . append ( hidden\_layer )

output\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_hidden  \+  1 ) \] }  for  i  in  range ( n\_outputs ) \]

network . append ( output\_layer )

 return network seed ( 1 )

network  \=  initialize\_network ( 2 ,  1 ,  2 )

for  layer in  network :

 print ( layer )

Running the example, you can see that the code prints out each layer one by one. You can see the hidden layer has one neuron with 2 input weights plus the bias. The output layer has 2 neurons, each with 1 weight plus the bias.

12

\[{'weights': \[0.13436424411240122, 0.8474337369372327, 0.763774618976614\]}\]

\[{'weights': \[0.2550690257394217, 0.49543508709194095\]}, {'weights': \[0.4494910647887381, 0.651592972722763\]}\]

Now that we know how to create and initialize a network, let’s see how we can use it to calculate an output.

### 2. Forward Propagate

We can calculate an output from a neural network by propagating an input signal through each layer until the output layer outputs its values.

We call this forward-propagation.

It is the technique we will need to generate predictions during training that will need to be corrected, and it is the method we will need after the network is trained to make predictions on new data.

We can break forward propagation down into three parts:

1. Neuron Activation.
2. Neuron Transfer.
3. Forward Propagation.

#### 2.1. Neuron Activation

The first step is to calculate the activation of one neuron given an input.

The input could be a row from our training dataset, as in the case of the hidden layer. It may also be the outputs from each neuron in the hidden layer, in the case of the output layer.

Neuron activation is calculated as the weighted sum of the inputs. Much like linear regression.

1

activation \= sum(weight\_i \* input\_i) \+ bias

Where **weight** is a network weight, **input** is an input, **i** is the index of a weight or an input and **bias** is a special weight that has no input to multiply with (or you can think of the input as always being 1.0).

Below is an implementation of this in a function named **activate()**. You can see that the function assumes that the bias is the last weight in the list of weights. This helps here and later to make the code easier to read.

1

2

3

4

5

6

\# Calculate neuron activation for an input

def activate ( weights ,  inputs ) :

activation  \=  weights \[ - 1 \]

for  i  in  range ( len ( weights ) - 1 ) :

activation  \+\=  weights \[ i \]  \*  inputs \[ i \]

 return activation

Now, let’s see how to use the neuron activation.

#### 2.2. Neuron Transfer

Once a neuron is activated, we need to transfer the activation to see what the neuron output actually is.

Different transfer functions can be used. It is traditional to use the [sigmoid activation function](https://en.wikipedia.org/wiki/Sigmoid_function), but you can also use the tanh ([hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_function)) function to transfer outputs. More recently, the [rectifier transfer function](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29) has been popular with large deep learning networks.

The sigmoid activation function looks like an S shape, it’s also called the logistic function. It can take any input value and produce a number between 0 and 1 on an S-curve. It is also a function of which we can easily calculate the derivative (slope) that we will need later when backpropagating error.

We can transfer an activation function using the sigmoid function as follows:

1

output \= 1 / (1 \+ e\^(-activation))

Where **e** is the base of the natural logarithms ([Euler’s number](https://en.wikipedia.org/wiki/E_%28mathematical_constant%29)).

Below is a function named **transfer()** that implements the sigmoid equation.

123

\# Transfer neuron activationdef transfer ( activation ) :

return  1.0  /  ( 1.0  \+  exp ( - activation ) )

Now that we have the pieces, let’s see how they are used.

#### 2.3. Forward Propagation

Forward propagating an input is straightforward.

We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer.

Below is a function named **forward\_propagate()** that implements the forward propagation for a row of data from our dataset with our neural network.

You can see that a neuron’s output value is stored in the neuron with the name ‘**output**‘. You can also see that we collect the outputs for a layer in an array named **new\_inputs** that becomes the array **inputs** and is used as inputs for the following layer.

The function returns the outputs from the last layer also called the output layer.

1

2

3

4

5

6

7

8

9

10

11

\# Forward propagate input to a network output

def forward\_propagate ( network ,  row ) :

 inputs \= row

for  layer in  network :

new\_inputs  \=  \[ \]

for  neuron in  layer :

activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )

neuron \[ 'output' \]  \=  transfer ( activation )

new\_inputs . append ( neuron \[ 'output' \] )

 inputs \= new\_inputs return inputs

Let’s put all of these pieces together and test out the forward propagation of our network.

We define our network inline with one hidden neuron that expects 2 input values and an output layer with two neurons.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

from math import exp # Calculate neuron activation for an input

def activate ( weights ,  inputs ) :

activation  \=  weights \[ - 1 \]

for  i  in  range ( len ( weights ) - 1 ) :

activation  \+\=  weights \[ i \]  \*  inputs \[ i \]

 return activation # Transfer neuron activationdef transfer ( activation ) :

return  1.0  /  ( 1.0  \+  exp ( - activation ) )

\# Forward propagate input to a network output

def forward\_propagate ( network ,  row ) :

 inputs \= row

for  layer in  network :

new\_inputs  \=  \[ \]

for  neuron in  layer :

activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )

neuron \[ 'output' \]  \=  transfer ( activation )

new\_inputs . append ( neuron \[ 'output' \] )

 inputs \= new\_inputs return inputs # test forward propagation

network  \=  \[ \[ { 'weights' :  \[ 0.13436424411240122 ,  0.8474337369372327 ,  0.763774618976614 \] } \] ,

\[ { 'weights' :  \[ 0.2550690257394217 ,  0.49543508709194095 \] } ,  { 'weights' :  \[ 0.4494910647887381 ,  0.651592972722763 \] } \] \]

row  \=  \[ 1 ,  0 ,  None \]

output  \=  forward\_propagate ( network ,  row )

print ( output )

Running the example propagates the input pattern \[1, 0\] and produces an output value that is printed. Because the output layer has two neurons, we get a list of two numbers as output.

The actual output values are just nonsense for now, but next, we will start to learn how to make the weights in the neurons more useful.

1

\[0.6629970129852887, 0.7253160725279748\]

### 3. Back Propagate Error

The backpropagation algorithm is named for the way in which weights are trained.

Error is calculated between the expected outputs and the outputs forward propagated from the network. These errors are then propagated backward through the network from the output layer to the hidden layer, assigning blame for the error and updating weights as they go.

The math for backpropagating error is rooted in calculus, but we will remain high level in this section and focus on what is calculated and how rather than why the calculations take this particular form.

This part is broken down into two sections.

1. Transfer Derivative.
2. Error Backpropagation.

#### 3.1. Transfer Derivative

Given an output value from a neuron, we need to calculate it’s slope.

We are using the sigmoid transfer function, the derivative of which can be calculated as follows:

1

derivative \= output \* (1.0 - output)

Below is a function named **transfer\_derivative()** that implements this equation.

123

\# Calculate the derivative of an neuron outputdef transfer\_derivative ( output ) :

return  output \*  ( 1.0  -  output )

Now, let’s see how this can be used.

#### 3.2. Error Backpropagation

The first step is to calculate the error for each output neuron, this will give us our error signal (input) to propagate backwards through the network.

The error for a given neuron can be calculated as follows:

1

error \= (output - expected) \* transfer\_derivative(output)

Where **expected** is the expected output value for the neuron, **output** is the output value for the neuron and **transfer\_derivative()** calculates the slope of the neuron’s output value, as shown above.

This error calculation is used for neurons in the output layer. The expected value is the class value itself. In the hidden layer, things are a little more complicated.

The error signal for a neuron in the hidden layer is calculated as the weighted error of each neuron in the output layer. Think of the error traveling back along the weights of the output layer to the neurons in the hidden layer.

The back-propagated error signal is accumulated and then used to determine the error for the neuron in the hidden layer, as follows:

1

error \= (weight\_k \* error\_j) \* transfer\_derivative(output)

Where **error\_j** is the error signal from the **j**th neuron in the output layer, **weight\_k** is the weight that connects the **k**th neuron to the current neuron and output is the output for the current neuron.

Below is a function named **backward\_propagate\_error()** that implements this procedure.

You can see that the error signal calculated for each neuron is stored with the name ‘delta’. You can see that the layers of the network are iterated in reverse order, starting at the output and working backwards. This ensures that the neurons in the output layer have ‘delta’ values calculated first that neurons in the hidden layer can use in the subsequent iteration. I chose the name ‘delta’ to reflect the change the error implies on the neuron (e.g. the weight delta).

You can see that the error signal for neurons in the hidden layer is accumulated from neurons in the output layer where the hidden neuron number **j** is also the index of the neuron’s weight in the output layer **neuron\[‘weights’\]\[j\]**.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

\# Backpropagate error and store in neurons

def backward\_propagate\_error ( network ,  expected ) :

for  i  in  reversed ( range ( len ( network ) ) ) :

layer  \=  network \[ i \]

errors  \=  list ( )

if  i  !\=  len ( network ) - 1 :

for  j  in  range ( len ( layer ) ) :

 error \= 0.0

for  neuron in  network \[ i  \+  1 \] :

error  \+\=  ( neuron \[ 'weights' \] \[ j \]  \*  neuron \[ 'delta' \] )

errors . append ( error )

 else :

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

errors . append ( neuron \[ 'output' \]  -  expected \[ j \] )

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

neuron \[ 'delta' \]  \=  errors \[ j \]  \*  transfer\_derivative ( neuron \[ 'output' \] )

Let’s put all of the pieces together and see how it works.

We define a fixed neural network with output values and backpropagate an expected output pattern. The complete example is listed below.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

\# Calculate the derivative of an neuron outputdef transfer\_derivative ( output ) :

return  output \*  ( 1.0  -  output )

\# Backpropagate error and store in neurons

def backward\_propagate\_error ( network ,  expected ) :

for  i  in  reversed ( range ( len ( network ) ) ) :

layer  \=  network \[ i \]

errors  \=  list ( )

if  i  !\=  len ( network ) - 1 :

for  j  in  range ( len ( layer ) ) :

 error \= 0.0

for  neuron in  network \[ i  \+  1 \] :

error  \+\=  ( neuron \[ 'weights' \] \[ j \]  \*  neuron \[ 'delta' \] )

errors . append ( error )

 else :

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

errors . append ( neuron \[ 'output' \]  -  expected \[ j \] )

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

neuron \[ 'delta' \]  \=  errors \[ j \]  \*  transfer\_derivative ( neuron \[ 'output' \] )

\# test backpropagation of error

network  \=  \[ \[ { 'output' :  0.7105668883115941 ,  'weights' :  \[ 0.13436424411240122 ,  0.8474337369372327 ,  0.763774618976614 \] } \] ,

\[ { 'output' :  0.6213859615555266 ,  'weights' :  \[ 0.2550690257394217 ,  0.49543508709194095 \] } ,  { 'output' :  0.6573693455986976 ,  'weights' :  \[ 0.4494910647887381 ,  0.651592972722763 \] } \] \]

expected  \=  \[ 0 ,  1 \]

backward\_propagate\_error ( network ,  expected )

for  layer in  network :

 print ( layer )

Running the example prints the network after the backpropagation of error is complete. You can see that error values are calculated and stored in the neurons for the output layer and the hidden layer.

12

\[{'output': 0.7105668883115941, 'weights': \[0.13436424411240122, 0.8474337369372327, 0.763774618976614\], 'delta': 0.0005348048046610517}\]

\[{'output': 0.6213859615555266, 'weights': \[0.2550690257394217, 0.49543508709194095\], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': \[0.4494910647887381, 0.651592972722763\], 'delta': -0.0771723774346327}\]

Now let’s use the backpropagation of error to train the network.

### 4. Train Network

The network is trained using stochastic gradient descent.

This involves multiple iterations of exposing a training dataset to the network and for each row of data forward propagating the inputs, backpropagating the error and updating the network weights.

This part is broken down into two sections:

1. Update Weights.
2. Train Network.

#### 4.1. Update Weights

Once errors are calculated for each neuron in the network via the back propagation method above, they can be used to update weights.

Network weights are updated as follows:

1

weight \= weight - learning\_rate \* error \* input

Where **weight** is a given weight, **learning\_rate** is a parameter that you must specify, **error** is the error calculated by the backpropagation procedure for the neuron and **input** is the input value that caused the error.

The same procedure can be used for updating the bias weight, except there is no input term, or input is the fixed value of 1.0.

Learning rate controls how much to change the weight to correct for the error. For example, a value of 0.1 will update the weight 10% of the amount that it possibly could be updated. Small learning rates are preferred that cause slower learning over a large number of training iterations. This increases the likelihood of the network finding a good set of weights across all layers rather than the fastest set of weights that minimize error (called premature convergence).

Below is a function named **update\_weights()** that updates the weights for a network given an input row of data, a learning rate and assume that a forward and backward propagation have already been performed.

Remember that the input for the output layer is a collection of outputs from the hidden layer.

1

2

3

4

5

6

7

8

9

10

\# Update network weights with error

def update\_weights ( network ,  row ,  l\_rate ) :

for  i  in  range ( len ( network ) ) :

inputs  \=  row \[ : - 1 \]

if  i  !\=  0 :

inputs  \=  \[ neuron \[ 'output' \]  for  neuron in  network \[ i  -  1 \] \]

for  neuron in  network \[ i \] :

for  j  in  range ( len ( inputs ) ) :

neuron \[ 'weights' \] \[ j \]  -\=  l\_rate \*  neuron \[ 'delta' \]  \*  inputs \[ j \]

neuron \[ 'weights' \] \[ - 1 \]  -\=  l\_rate \*  neuron \[ 'delta' \]

Now we know how to update network weights, let’s see how we can do it repeatedly.

#### 4.2. Train Network

As mentioned, the network is updated using stochastic gradient descent.

This involves first looping for a fixed number of epochs and within each epoch updating the network for each row in the training dataset.

Because updates are made for each training pattern, this type of learning is called online learning. If errors were accumulated across an epoch before updating the weights, this is called batch learning or batch gradient descent.

Below is a function that implements the training of an already initialized neural network with a given training dataset, learning rate, fixed number of epochs and an expected number of output values.

The expected number of output values is used to transform class values in the training data into a one hot encoding. That is a binary vector with one column for each class value to match the output of the network. This is required to calculate the error for the output layer.

You can also see that the sum squared error between the expected output and the network output is accumulated each epoch and printed. This is helpful to create a trace of how much the network is learning and improving each epoch.

1

2

3

4

5

6

7

8

9

10

11

12

\# Train a network for a fixed number of epochs

def train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs ) :

for  epoch in  range ( n\_epoch ) :

 sum\_error \= 0

for  row in  train :

outputs  \=  forward\_propagate ( network ,  row )

expected  \=  \[ 0  for  i  in  range ( n\_outputs ) \]

expected \[ row \[ - 1 \] \]  \=  1

sum\_error  \+\=  sum ( \[ ( expected \[ i \] - outputs \[ i \] ) \* \* 2  for  i  in  range ( len ( expected ) ) \] )

backward\_propagate\_error ( network ,  expected )

update\_weights ( network ,  row ,  l\_rate )

print ( '>epoch\=%d, lrate\=%.3f, error\=%.3f'  %  ( epoch ,  l\_rate ,  sum\_error ) )

We now have all of the pieces to train the network. We can put together an example that includes everything we’ve seen so far including network initialization and train a network on a small dataset.

Below is a small contrived dataset that we can use to test out training our neural network.

1

2

3

4

5

6

7

8

9

10

11

X1 X2 Y

2.7810836 2.550537003 0

1.465489372 2.362125076 0

3.396561688 4.400293529 0

1.38807019 1.850220317 0

3.06407232 3.005305973 0

7.627531214 2.759262235 1

5.332441248 2.088626775 1

6.922596716 1.77106367 1

8.675418651 -0.242068655 1

7.673756466 3.508563011 1

Below is the complete example. We will use 2 neurons in the hidden layer. It is a binary classification problem (2 classes) so there will be two neurons in the output layer. The network will be trained for 20 epochs with a learning rate of 0.5, which is high because we are training for so few iterations.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

from math import exp

from random import seed

from random import random

\# Initialize a network

def initialize\_network ( n\_inputs ,  n\_hidden ,  n\_outputs ) :

network  \=  list ( )

hidden\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_inputs  \+  1 ) \] }  for  i  in  range ( n\_hidden ) \]

network . append ( hidden\_layer )

output\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_hidden  \+  1 ) \] }  for  i  in  range ( n\_outputs ) \]

network . append ( output\_layer )

 return network # Calculate neuron activation for an input

def activate ( weights ,  inputs ) :

activation  \=  weights \[ - 1 \]

for  i  in  range ( len ( weights ) - 1 ) :

activation  \+\=  weights \[ i \]  \*  inputs \[ i \]

 return activation # Transfer neuron activationdef transfer ( activation ) :

return  1.0  /  ( 1.0  \+  exp ( - activation ) )

\# Forward propagate input to a network output

def forward\_propagate ( network ,  row ) :

 inputs \= row

for  layer in  network :

new\_inputs  \=  \[ \]

for  neuron in  layer :

activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )

neuron \[ 'output' \]  \=  transfer ( activation )

new\_inputs . append ( neuron \[ 'output' \] )

inputs  \=  new\_inputs

return  inputs

\# Calculate the derivative of an neuron output

def transfer\_derivative ( output ) :

return  output \*  ( 1.0  -  output )

\# Backpropagate error and store in neurons

def backward\_propagate\_error ( network ,  expected ) :

for  i  in  reversed ( range ( len ( network ) ) ) :

layer  \=  network \[ i \]

errors  \=  list ( )

if  i  !\=  len ( network ) - 1 :

for  j  in  range ( len ( layer ) ) :

 error \= 0.0

for  neuron in  network \[ i  \+  1 \] :

error  \+\=  ( neuron \[ 'weights' \] \[ j \]  \*  neuron \[ 'delta' \] )

errors . append ( error )

 else :

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

errors . append ( neuron \[ 'output' \]  -  expected \[ j \] )

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

neuron \[ 'delta' \]  \=  errors \[ j \]  \*  transfer\_derivative ( neuron \[ 'output' \] )

\# Update network weights with error

def update\_weights ( network ,  row ,  l\_rate ) :

for  i  in  range ( len ( network ) ) :

inputs  \=  row \[ : - 1 \]

if  i  !\=  0 :

inputs  \=  \[ neuron \[ 'output' \]  for  neuron in  network \[ i  -  1 \] \]

for  neuron in  network \[ i \] :

for  j  in  range ( len ( inputs ) ) :

neuron \[ 'weights' \] \[ j \]  -\=  l\_rate \*  neuron \[ 'delta' \]  \*  inputs \[ j \]

neuron \[ 'weights' \] \[ - 1 \]  -\=  l\_rate \*  neuron \[ 'delta' \]

\# Train a network for a fixed number of epochs

def train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs ) :

for  epoch in  range ( n\_epoch ) :

 sum\_error \= 0

for  row in  train :

outputs  \=  forward\_propagate ( network ,  row )

expected  \=  \[ 0  for  i  in  range ( n\_outputs ) \]

expected \[ row \[ - 1 \] \]  \=  1

sum\_error  \+\=  sum ( \[ ( expected \[ i \] - outputs \[ i \] ) \* \* 2  for  i  in  range ( len ( expected ) ) \] )

backward\_propagate\_error ( network ,  expected )

update\_weights ( network ,  row ,  l\_rate )

print ( '>epoch\=%d, lrate\=%.3f, error\=%.3f'  %  ( epoch ,  l\_rate ,  sum\_error ) )

\# Test training backprop algorithmseed ( 1 )

dataset  \=  \[ \[ 2.7810836 , 2.550537003 , 0 \] ,

\[ 1.465489372 , 2.362125076 , 0 \] ,

\[ 3.396561688 , 4.400293529 , 0 \] ,

\[ 1.38807019 , 1.850220317 , 0 \] ,

\[ 3.06407232 , 3.005305973 , 0 \] ,

\[ 7.627531214 , 2.759262235 , 1 \] ,

\[ 5.332441248 , 2.088626775 , 1 \] ,

\[ 6.922596716 , 1.77106367 , 1 \] ,

\[ 8.675418651 , - 0.242068655 , 1 \] ,

\[ 7.673756466 , 3.508563011 , 1 \] \]

n\_inputs  \=  len ( dataset \[ 0 \] )  -  1

n\_outputs  \=  len ( set ( \[ row \[ - 1 \]  for  row in  dataset \] ) )

network  \=  initialize\_network ( n\_inputs ,  2 ,  n\_outputs )

train\_network ( network ,  dataset ,  0.5 ,  20 ,  n\_outputs )

for  layer in  network :

 print ( layer )

Running the example first prints the sum squared error each training epoch. We can see a trend of this error decreasing with each epoch.

Once trained, the network is printed, showing the learned weights. Also still in the network are output and delta values that can be ignored. We could update our training function to delete these data if we wanted.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

\>epoch\=0, lrate\=0.500, error\=6.350

\>epoch\=1, lrate\=0.500, error\=5.531

\>epoch\=2, lrate\=0.500, error\=5.221

\>epoch\=3, lrate\=0.500, error\=4.951

\>epoch\=4, lrate\=0.500, error\=4.519

\>epoch\=5, lrate\=0.500, error\=4.173

\>epoch\=6, lrate\=0.500, error\=3.835

\>epoch\=7, lrate\=0.500, error\=3.506

\>epoch\=8, lrate\=0.500, error\=3.192

\>epoch\=9, lrate\=0.500, error\=2.898

\>epoch\=10, lrate\=0.500, error\=2.626

\>epoch\=11, lrate\=0.500, error\=2.377

\>epoch\=12, lrate\=0.500, error\=2.153

\>epoch\=13, lrate\=0.500, error\=1.953

\>epoch\=14, lrate\=0.500, error\=1.774

\>epoch\=15, lrate\=0.500, error\=1.614

\>epoch\=16, lrate\=0.500, error\=1.472

\>epoch\=17, lrate\=0.500, error\=1.346

\>epoch\=18, lrate\=0.500, error\=1.233

\>epoch\=19, lrate\=0.500, error\=1.132

\[{'weights': \[-1.4688375095432327, 1.850887325439514, 1.0858178629550297\], 'output': 0.029980305604426185, 'delta': 0.0059546604162323625}, {'weights': \[0.37711098142462157, -0.0625909894552989, 0.2765123702642716\], 'output': 0.9456229000211323, 'delta': -0.0026279652850863837}\]

\[{'weights': \[2.515394649397849, -0.3391927502445985, -0.9671565426390275\], 'output': 0.23648794202357587, 'delta': 0.04270059278364587}, {'weights': \[-2.5584149848484263, 1.0036422106209202, 0.42383086467582715\], 'output': 0.7790535202438367, 'delta': -0.03803132596437354}\]

Once a network is trained, we need to use it to make predictions.

### 5. Predict

Making predictions with a trained neural network is easy enough.

We have already seen how to forward-propagate an input pattern to get an output. This is all we need to do to make a prediction. We can use the output values themselves directly as the probability of a pattern belonging to each output class.

It may be more useful to turn this output back into a crisp class prediction. We can do this by selecting the class value with the larger probability. This is also called the [arg max function](https://en.wikipedia.org/wiki/Arg_max).

Below is a function named **predict()** that implements this procedure. It returns the index in the network output that has the largest probability. It assumes that class values have been converted to integers starting at 0.

1234

\# Make a prediction with a network

def predict ( network ,  row ) :

outputs  \=  forward\_propagate ( network ,  row )

return  outputs . index ( max ( outputs ) )

We can put this together with our code above for forward propagating input and with our small contrived dataset to test making predictions with an already-trained network. The example hardcodes a network trained from the previous step.

The complete example is listed below.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

from math import exp # Calculate neuron activation for an input

def activate ( weights ,  inputs ) :

activation  \=  weights \[ - 1 \]

for  i  in  range ( len ( weights ) - 1 ) :

activation  \+\=  weights \[ i \]  \*  inputs \[ i \]

 return activation # Transfer neuron activationdef transfer ( activation ) :

return  1.0  /  ( 1.0  \+  exp ( - activation ) )

\# Forward propagate input to a network output

def forward\_propagate ( network ,  row ) :

 inputs \= row

for  layer in  network :

new\_inputs  \=  \[ \]

for  neuron in  layer :

activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )

neuron \[ 'output' \]  \=  transfer ( activation )

new\_inputs . append ( neuron \[ 'output' \] )

 inputs \= new\_inputs return inputs # Make a prediction with a network

def predict ( network ,  row ) :

outputs  \=  forward\_propagate ( network ,  row )

return  outputs . index ( max ( outputs ) )

\# Test making predictions with the network

dataset  \=  \[ \[ 2.7810836 , 2.550537003 , 0 \] ,

\[ 1.465489372 , 2.362125076 , 0 \] ,

\[ 3.396561688 , 4.400293529 , 0 \] ,

\[ 1.38807019 , 1.850220317 , 0 \] ,

\[ 3.06407232 , 3.005305973 , 0 \] ,

\[ 7.627531214 , 2.759262235 , 1 \] ,

\[ 5.332441248 , 2.088626775 , 1 \] ,

\[ 6.922596716 , 1.77106367 , 1 \] ,

\[ 8.675418651 , - 0.242068655 , 1 \] ,

\[ 7.673756466 , 3.508563011 , 1 \] \]

network  \=  \[ \[ { 'weights' :  \[ - 1.482313569067226 ,  1.8308790073202204 ,  1.078381922048799 \] } ,  { 'weights' :  \[ 0.23244990332399884 ,  0.3621998343835864 ,  0.40289821191094327 \] } \] ,

\[ { 'weights' :  \[ 2.5001872433501404 ,  0.7887233511355132 ,  - 1.1026649757805829 \] } ,  { 'weights' :  \[ - 2.429350576245497 ,  0.8357651039198697 ,  1.0699217181280656 \] } \] \]

for  row in  dataset :

prediction  \=  predict ( network ,  row )

print ( 'Expected\=%d, Got\=%d'  %  ( row \[ - 1 \] ,  prediction ) )

Running the example prints the expected output for each record in the training dataset, followed by the crisp prediction made by the network.

It shows that the network achieves 100% accuracy on this small dataset.

1

2

3

4

5

6

7

8

9

10

Expected\=0, Got\=0

Expected\=0, Got\=0

Expected\=0, Got\=0

Expected\=0, Got\=0

Expected\=0, Got\=0

Expected\=1, Got\=1

Expected\=1, Got\=1

Expected\=1, Got\=1

Expected\=1, Got\=1

Expected\=1, Got\=1

Now we are ready to apply our backpropagation algorithm to a real world dataset.

### 6. Wheat Seeds Dataset

This section applies the Backpropagation algorithm to the wheat seeds dataset.

The first step is to load the dataset and convert the loaded data to numbers that we can use in our neural network. For this we will use the helper function **load\_csv()** to load the file, **str\_column\_to\_float()** to convert string numbers to floats and **str\_column\_to\_int()** to convert the class column to integer values.

Input values vary in scale and need to be normalized to the range of 0 and 1. It is generally good practice to normalize input values to the range of the chosen transfer function, in this case, the sigmoid function that outputs values between 0 and 1. The **dataset\_minmax()** and **normalize\_dataset()** helper functions were used to normalize the input values.

We will evaluate the algorithm using k-fold cross-validation with 5 folds. This means that 201/5\=40.2 or 40 records will be in each fold. We will use the helper functions **evaluate\_algorithm()** to evaluate the algorithm with cross-validation and **accuracy\_metric()** to calculate the accuracy of predictions.

A new function named **back\_propagation()** was developed to manage the application of the Backpropagation algorithm, first initializing a network, training it on the training dataset and then using the trained network to make predictions on a test dataset.

The complete example is listed below.

1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

36

37

38

39

40

41

42

43

44

45

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

78

79

80

81

82

83

84

85

86

87

88

89

90

91

92

93

94

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

162

163

164

165

166

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

187

188

189

190

191

192

193

194

195

196

197

198

199

\# Backprop on the Seeds Dataset

from random import seed

from random import randrange

from random import random

from csv import reader

from math import exp

\# Load a CSV file

def load\_csv ( filename ) :

dataset  \=  list ( )

with open ( filename ,  'r' )  as  file :

csv\_reader  \=  reader ( file )

for  row in  csv\_reader :

if  not  row :

 continue

dataset . append ( row )

 return dataset # Convert string column to float

def str\_column\_to\_float ( dataset ,  column ) :

for  row in  dataset :

row \[ column \]  \=  float ( row \[ column \] . strip ( ) )

\# Convert string column to integer

def str\_column\_to\_int ( dataset ,  column ) :

class\_values  \=  \[ row \[ column \]  for  row in  dataset \]

unique  \=  set ( class\_values )

lookup  \=  dict ( )

for  i ,  value in  enumerate ( unique ) :

lookup \[ value \]  \=  i

for  row in  dataset :

row \[ column \]  \=  lookup \[ row \[ column \] \]

 return lookup # Find the min and max values for each columndef dataset\_minmax ( dataset ) :

minmax  \=  list ( )

stats  \=  \[ \[ min ( column ) ,  max ( column ) \]  for  column in  zip ( \* dataset ) \]

 return stats # Rescale dataset columns to the range 0-1

def normalize\_dataset ( dataset ,  minmax ) :

for  row in  dataset :

for  i  in  range ( len ( row ) - 1 ) :

row \[ i \]  \=  ( row \[ i \]  -  minmax \[ i \] \[ 0 \] )  /  ( minmax \[ i \] \[ 1 \]  -  minmax \[ i \] \[ 0 \] )

\# Split a dataset into k folds

def cross\_validation\_split ( dataset ,  n\_folds ) :

dataset\_split  \=  list ( )

dataset\_copy  \=  list ( dataset )

fold\_size  \=  int ( len ( dataset )  /  n\_folds )

for  i  in  range ( n\_folds ) :

fold  \=  list ( )

while  len ( fold )  \<  fold\_size :

index  \=  randrange ( len ( dataset\_copy ) )

fold . append ( dataset\_copy . pop ( index ) )

dataset\_split . append ( fold )

 return dataset \_split # Calculate accuracy percentage

def accuracy\_metric ( actual ,  predicted ) :

 correct \= 0

for  i  in  range ( len ( actual ) ) :

if  actual \[ i \]  \=\=  predicted \[ i \] :

 correct \+\= 1

return  correct  /  float ( len ( actual ) )  \*  100.0

\# Evaluate an algorithm using a cross validation split

def evaluate\_algorithm ( dataset ,  algorithm ,  n\_folds ,  \* args ) :

folds  \=  cross\_validation\_split ( dataset ,  n\_folds )

scores  \=  list ( )

for  fold in  folds :

train\_set  \=  list ( folds )

train\_set . remove ( fold )

train\_set  \=  sum ( train\_set ,  \[ \] )

test\_set  \=  list ( )

for  row in  fold :

row\_copy  \=  list ( row )

test\_set . append ( row\_copy )

row\_copy \[ - 1 \]  \=  None

predicted  \=  algorithm ( train\_set ,  test\_set ,  \* args )

actual  \=  \[ row \[ - 1 \]  for  row in  fold \]

accuracy  \=  accuracy\_metric ( actual ,  predicted )

scores . append ( accuracy )

 return scores # Calculate neuron activation for an input

def activate ( weights ,  inputs ) :

activation  \=  weights \[ - 1 \]

for  i  in  range ( len ( weights ) - 1 ) :

activation  \+\=  weights \[ i \]  \*  inputs \[ i \]

 return activation # Transfer neuron activationdef transfer ( activation ) :

return  1.0  /  ( 1.0  \+  exp ( - activation ) )

\# Forward propagate input to a network output

def forward\_propagate ( network ,  row ) :

 inputs \= row

for  layer in  network :

new\_inputs  \=  \[ \]

for  neuron in  layer :

activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )

neuron \[ 'output' \]  \=  transfer ( activation )

new\_inputs . append ( neuron \[ 'output' \] )

inputs  \=  new\_inputs

return  inputs

\# Calculate the derivative of an neuron output

def transfer\_derivative ( output ) :

return  output \*  ( 1.0  -  output )

\# Backpropagate error and store in neurons

def backward\_propagate\_error ( network ,  expected ) :

for  i  in  reversed ( range ( len ( network ) ) ) :

layer  \=  network \[ i \]

errors  \=  list ( )

if  i  !\=  len ( network ) - 1 :

for  j  in  range ( len ( layer ) ) :

 error \= 0.0

for  neuron in  network \[ i  \+  1 \] :

error  \+\=  ( neuron \[ 'weights' \] \[ j \]  \*  neuron \[ 'delta' \] )

errors . append ( error )

 else :

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

errors . append ( neuron \[ 'output' \]  -  expected \[ j \] )

for  j  in  range ( len ( layer ) ) :

neuron  \=  layer \[ j \]

neuron \[ 'delta' \]  \=  errors \[ j \]  \*  transfer\_derivative ( neuron \[ 'output' \] )

\# Update network weights with error

def update\_weights ( network ,  row ,  l\_rate ) :

for  i  in  range ( len ( network ) ) :

inputs  \=  row \[ : - 1 \]

if  i  !\=  0 :

inputs  \=  \[ neuron \[ 'output' \]  for  neuron in  network \[ i  -  1 \] \]

for  neuron in  network \[ i \] :

for  j  in  range ( len ( inputs ) ) :

neuron \[ 'weights' \] \[ j \]  -\=  l\_rate \*  neuron \[ 'delta' \]  \*  inputs \[ j \]

neuron \[ 'weights' \] \[ - 1 \]  -\=  l\_rate \*  neuron \[ 'delta' \]

\# Train a network for a fixed number of epochs

def train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs ) :

for  epoch in  range ( n\_epoch ) :

for  row in  train :

outputs  \=  forward\_propagate ( network ,  row )

expected  \=  \[ 0  for  i  in  range ( n\_outputs ) \]

expected \[ row \[ - 1 \] \]  \=  1

backward\_propagate\_error ( network ,  expected )

update\_weights ( network ,  row ,  l\_rate )

\# Initialize a network

def initialize\_network ( n\_inputs ,  n\_hidden ,  n\_outputs ) :

network  \=  list ( )

hidden\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_inputs  \+  1 ) \] }  for  i  in  range ( n\_hidden ) \]

network . append ( hidden\_layer )

output\_layer  \=  \[ { 'weights' : \[ random ( )  for  i  in  range ( n\_hidden  \+  1 ) \] }  for  i  in  range ( n\_outputs ) \]

network . append ( output\_layer )

 return network # Make a prediction with a network

def predict ( network ,  row ) :

outputs  \=  forward\_propagate ( network ,  row )

return  outputs . index ( max ( outputs ) )

\# Backpropagation Algorithm With Stochastic Gradient Descent

def back\_propagation ( train ,  test ,  l\_rate ,  n\_epoch ,  n\_hidden ) :

n\_inputs  \=  len ( train \[ 0 \] )  -  1

n\_outputs  \=  len ( set ( \[ row \[ - 1 \]  for  row in  train \] ) )

network  \=  initialize\_network ( n\_inputs ,  n\_hidden ,  n\_outputs )

train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs )

predictions  \=  list ( )

for  row in  test :

prediction  \=  predict ( network ,  row )

predictions . append ( prediction )

return ( predictions )

\# Test Backprop on Seeds dataset

seed ( 1 )

\# load and prepare data

filename  \=  'seeds\_dataset.csv'

dataset  \=  load\_csv ( filename )

for  i  in  range ( len ( dataset \[ 0 \] ) - 1 ) :

str\_column\_to\_float ( dataset ,  i )

\# convert class column to integers

str\_column\_to\_int ( dataset ,  len ( dataset \[ 0 \] ) - 1 )

\# normalize input variables

minmax  \=  dataset\_minmax ( dataset )

normalize\_dataset ( dataset ,  minmax )

\# evaluate algorithm

n\_folds  \=  5

l\_rate  \=  0.3

n\_epoch  \=  500

n\_hidden  \=  5

scores  \=  evaluate\_algorithm ( dataset ,  back\_propagation ,  n\_folds ,  l\_rate ,  n\_epoch ,  n\_hidden )

print ( 'Scores: %s'  %  scores )

print ( 'Mean Accuracy: %.3f%%'  %  ( sum ( scores ) / float ( len ( scores ) ) ) )

A network with 5 neurons in the hidden layer and 3 neurons in the output layer was constructed. The network was trained for 500 epochs with a learning rate of 0.3. These parameters were found with a little trial and error, but you may be able to do much better.

Running the example prints the average classification accuracy on each fold as well as the average performance across all folds.

You can see that backpropagation and the chosen configuration achieved a mean classification accuracy of about 93% which is dramatically better than the Zero Rule algorithm that did slightly better than 28% accuracy.

12

Scores: \[92.85714285714286, 92.85714285714286, 97.61904761904762, 92.85714285714286, 90.47619047619048\]

Mean Accuracy: 93.333%

## Extensions

This section lists extensions to the tutorial that you may wish to explore.

- **Tune Algorithm Parameters**. Try larger or smaller networks trained for longer or shorter. See if you can get better performance on the seeds dataset.
- **Additional Methods**. Experiment with different weight initialization techniques (such as small random numbers) and different transfer functions (such as tanh).
- **More Layers**. Add support for more hidden layers, trained in just the same way as the one hidden layer used in this tutorial.
- **Regression**. Change the network so that there is only one neuron in the output layer and that a real value is predicted. Pick a regression dataset to practice on. A linear transfer function could be used for neurons in the output layer, or the output values of the chosen dataset could be scaled to values between 0 and 1.
- **Batch Gradient Descent**. Change the training procedure from online to batch gradient descent and update the weights only at the end of each epoch.

**Did you try any of these extensions?** \
 Share your experiences in the comments below.

## Review

In this tutorial, you discovered how to implement the Backpropagation algorithm from scratch.

Specifically, you learned:

- How to forward propagate an input to calculate a network output.
- How to back propagate error and update network weights.
- How to apply the backpropagation algorithm to a real world dataset.

**Do you have any questions?** \
 Ask your questions in the comments below and I will do my best to answer.

## Discover How to Code Algorithms From Scratch!

[![Machine Learning Algorithms From Scratch](https://wp-content/uploads/2022/11/MMLA-220.png)](https://machinelearningmastery.com/machine-learning-algorithms-from-scratch/)

#### No Libraries, Just Python Code.

...with step-by-step tutorials on real-world datasets

Discover how in my new Ebook:\
 [Machine Learning Algorithms From Scratch](https://machinelearningmastery.com/machine-learning-algorithms-from-scratch/)

It covers **18 tutorials** with all the code for **12 top algorithms**, like:\
 Linear Regression, k-Nearest Neighbors, Stochastic Gradient Descent and much more...

#### Finally, Pull Back the Curtain on\
 Machine Learning Algorithms

Skip the Academics. Just Results.

[See What's Inside](https://machinelearningmastery.com/machine-learning-algorithms-from-scratch/)

### More On This Topic

- [![8 Tricks for Configuring Backpropagation to Train Better Neural Networks, Faster](https://machinelearningmastery.com/wp-content/uploads/2019/02/8-Tricks-for-Configuring-Backpropagation-to-Train-Better-Neural-Networks-Faster.jpg "8 Tricks for Configuring Backpropagation to Train Better Neural Networks"){width=200 height=133} 8 Tricks for Configuring Backpropagation to Train…](https://machinelearningmastery.com/best-advice-for-configuring-backpropagation-for-deep-learning-neural-networks/)
- [![Difference Between Backpropagation and Stochastic Gradient Descent](https://machinelearningmastery.com/wp-content/uploads/2021/05/Difference-Between-Backpropagation-and-Stochastic-Gradient-Descent.jpg "Difference Between Backpropagation and Stochastic Gradient Descent"){width=200 height=133} Difference Between Backpropagation and Stochastic…](https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/)
- [![A Gentle Introduction to Backpropagation Through Time](https://machinelearningmastery.com/wp-content/uploads/2017/06/A-Gentle-Introduction-to-Backpropagation-Through-Time.jpg "A Gentle Introduction to Backpropagation Through Time"){width=200 height=134} A Gentle Introduction to Backpropagation Through Time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
- [![Encoder-Decoder Recurrent Neural Network Models for Neural Machine Translation](https://machinelearningmastery.com/wp-content/uploads/2018/01/Encoder-Decoder-Recurrent-Neural-Network-Models-for-Neural-Machine-Translation.jpg "Encoder-Decoder Recurrent Neural Network Models for Neural Machine Translation"){width=200 height=113} Encoder-Decoder Recurrent Neural Network Models for…](https://machinelearningmastery.com/encoder-decoder-recurrent-neural-network-models-neural-machine-translation/)
- [![How to Code the Student's t-Test from Scratch in Python](https://machinelearningmastery.com/wp-content/uploads/2018/07/How-to-Code-the-Students-t-Test-from-Scratch-in-Python.jpg "How to Code the Student's t-Test from Scratch in Python"){width=200 height=133} How to Code the Student's t-Test from Scratch in Python](https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/)
- [![Contour Plot of the Test Objective Function With Adam Search Results Shown](https://machinelearningmastery.com/wp-content/uploads/2020/12/Contour-Plot-of-the-Test-Objective-Function-With-Adam-Search-Results-Shown.png "Code Adam Optimization Algorithm From Scratch"){width=200 height=150} Code Adam Optimization Algorithm From Scratch](https://machinelearningmastery.com/adam-optimization-from-scratch/)

[ How To Implement Learning Vector Quantization (LVQ) From Scratch With Python](https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/)

[How To Implement The Decision Tree Algorithm From Scratch In Python ](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)

### 845 Responses to *How to Code a Neural Network with Backpropagation In Python (from scratch)* {#comments-title}

1. 
[Talk Data To Me](https://www.facebook.com/talkdatatome/) November 7, 2016 at 9:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-369793 "Direct link to this comment")That’s what I was looking for. Write a neural network without any libraries (scikit, keras etc.) Thnak you very much!    - 
[Jason Brownlee](https://machinelearningmastery.com) November 8, 2016 at 9:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-369865 "Direct link to this comment")I’m glad to hear it!        - 
[sari dewi](http://saridewi.web.ugm.ac.id) August 16, 2019 at 11:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496972 "Direct link to this comment")Hy Mr. jason , i try your code to make a neural network with backpropagation method, I using jupyter notebook anaconda and pyhton 3.7 64 bit, when i try this codeseed(1)\
 # load and prepare data\
 filename \=’datalatih.csv’\
 dataset \= load\_csv(filename)\
 for i in range(len(dataset\[0\])-1):\
 str\_column\_to\_float(dataset, i)\
 # convert class column to integers\
 str\_column\_to\_int(dataset, len(dataset\[0\])-1)\
 # normalize input variables\
 minmax \= dataset\_minmax(dataset)\
 normalize\_dataset(dataset, minmax)\
 # evaluate algorithm\
 n\_folds \=5\
 l\_rate \=0.3\
 n\_epoch \=500\
 n\_hidden \=5\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)print (‘Scores: %s’ % scores)\
 print (‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores))))but I get error messageIndexError Traceback (most recent call last)\
 in\
 196 n\_epoch \=500\
 197 n\_hidden \=5\
 –> 198 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 199\
 200 print (‘Scores: %s’ % scores) in evaluate\_algorithm(dataset, algorithm, n\_folds, \*args)\
 79 test\_set.append(row\_copy)\
 80 row\_copy\[-1\] \= None\
 —> 81 predicted \= algorithm(train\_set, test\_set, \*args)\
 82 actual \= \[row\[-1\] for row in fold\]\
 83 accuracy \= accuracy\_metric(actual, predicted) in back\_propagation(train, test, l\_rate, n\_epoch, n\_hidden)\
 171 n\_outputs \= len(set(\[row\[-1\] for row in train\]))\
 172 network \= initialize\_network(n\_inputs, n\_hidden, n\_outputs)\
 –> 173 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 174 predictions \= list()\
 175 for row in test: in train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 148 outputs \= forward\_propagate(network, row)\
 149 expected \= \[0 for i in range(n\_outputs)\]\
 –> 150 expected\[row\[-1\]\] \= 1\
 151 backward\_propagate\_error(network, expected)\
 152 update\_weights(network, row, l\_rate)IndexError: list assignment index out of rangewhat my mistake? is there missing code? thankyou            - 
[Jason Brownlee](https://machinelearningmastery.com) August 16, 2019 at 2:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496988 "Direct link to this comment")Sorry to hear that you are having trouble, I have some suggestions for you here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)            - 
steven November 9, 2019 at 1:35 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-509358 "Direct link to this comment")this is the exact problem I face to. Do you have any suggestion? Thank you so much        - 
[Febry Triyadi](http://not%20yet) November 22, 2019 at 6:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-511941 "Direct link to this comment")Hi Mr.Jason i have trouble with your code. Please check it, i not understand with expected\[row\[-1\]\] \= 1IndexError Traceback (most recent call last)\
 in ()\
 13 n\_epoch \= 500\
 14 n\_hidden \= 5\
 —> 15 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 16 print(‘Scores: %s’ % scores)\
 17 print(‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores))))2 frames\
 in train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 50 outputs \= forward\_propagate(network, row)\
 51 expected \= \[0 for i in range(n\_outputs)\]\
 —> 52 expected\[row\[-1\]\] \= 1\
 53 backward\_propagate\_error(network, expected)\
 54 update\_weights(network, row, l\_rate)IndexError: list assignment index out of range            - 
[Jason Brownlee](https://machinelearningmastery.com) November 23, 2019 at 6:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-511987 "Direct link to this comment")Sorry to hear that, I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)    - 
[WB](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) February 20, 2018 at 3:07 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430007 "Direct link to this comment")I experienced the following applying the Backpropagation algorithm to the wheat seeds dataset. I am wondering how to resolve the errors? Thank you\
 —————————————————————————\
 ValueError Traceback (most recent call last)\
 in ()\
 184 dataset \= load\_csv(filename)\
 185 for i in range(len(dataset\[0\])-1):\
 –> 186 str\_column\_to\_float(dataset, i)\
 187 # convert class column to integers\
 188 str\_column\_to\_int(dataset, len(dataset\[0\])-1) in str\_column\_to\_float(dataset, column)\
 20 def str\_column\_to\_float(dataset, column):\
 21 for row in dataset:\
 —> 22 row\[column\] \= float(row\[column\].strip())\
 23\
 24 # Convert string column to integerValueError: could not convert string to float:        - 
[Jason Brownlee](https://machinelearningmastery.com) February 21, 2018 at 6:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430095 "Direct link to this comment")Are you using Python 2?            - 
[wb](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430007) February 21, 2018 at 2:51 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430134 "Direct link to this comment")Yes I am            - 
harshith October 5, 2018 at 8:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-450776 "Direct link to this comment")hi bro whass up        - 
Mike Harney March 5, 2018 at 9:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431188 "Direct link to this comment")Hi wb, I’m on 3.6 and I found the same issue. Maybe you can answer this Jason, but it looks like the some of the data is misaligned in the sample. When opened in Excel, there are many open spaces followed by data jutted out to an extra column. I assume this is unintentional, and when I corrected the spacing, it appeared to work for me.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 6, 2018 at 6:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431265 "Direct link to this comment")The code was written and tested with Python 2.7.                - 
JU April 23, 2018 at 7:24 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435683 "Direct link to this comment")Mike is right – the dataset from the UCI website is slightly defective: It has two tabs in some places where there should be only one. This needs to be corrected during the conversion to CSV. In Excel the easiest way is to use the text importer and then click the “Treat consecutive delimiters as one” checkbox.                - 
[Jason Brownlee](https://machinelearningmastery.com) April 23, 2018 at 7:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435685 "Direct link to this comment")Here is the dataset ready to use:\
 [https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)        - 
Alexis Batyk August 29, 2018 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-447163 "Direct link to this comment")\[SOLVED\]\
 i have the same issue with [https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv)there is still dirty that csvuse a text editor -> select search and replace tool -> search ‘,,’ replace ‘,’ and it works            - 
[Jason Brownlee](https://machinelearningmastery.com) August 29, 2018 at 8:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-447193 "Direct link to this comment")I don’t have such problems on Py 3.6.            - 
[Jackson Scott](http://jacksonscott.net) October 1, 2018 at 9:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-450367 "Direct link to this comment")thanks, this worked for me as well. The csv file had some tabbed over and others correct.                - 
Dharmendra Kumar September 3, 2019 at 7:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-499531 "Direct link to this comment")Thank you                - 
[Jason Brownlee](https://machinelearningmastery.com) September 4, 2019 at 5:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-499593 "Direct link to this comment")You’re welcome.        - 
Deng October 14, 2018 at 5:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451606 "Direct link to this comment")The data in the seeds\_dataset file contains the backspace key, and it is ok to reset the data    - 
[George Dong](http://homeschoolworld.co.uk) May 12, 2019 at 6:14 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-485046 "Direct link to this comment")I echo that too!Just one question please! In your code below, I could not understand why multiplication is used instead of division in the last line. Though division caused divide by zero problem.1234567891011121314151617181920212223# Backpropagate error and store in neuronsdef backward\_propagate\_error ( network ,  expected ) :for  i  in  reversed ( range ( len ( network ) ) ) :layer  \=  network \[ i \]errors  \=  list ( )if  i  !\=  len ( network ) - 1 :for  j  in  range ( len ( layer ) ) : error \= 0.0for  neuron in  network \[ i  \+  1 \] :error  \+\=  ( neuron \[ 'weights' \] \[ j \]  \*  neuron \[ 'delta' \] )errors . append ( error )for  j  in  range ( len ( layer ) ) :layer \[ j \] \[ 'error' \]  \=  errors \[ j \] else :for  j  in  range ( len ( layer ) ) :neuron  \=  layer \[ j \]errors . append ( expected \[ j \]  -  neuron \[ 'output' \] )neuron \[ 'error' \] \= expected \[ j \]  -  neuron \[ 'output' \]for  j  in  range ( len ( layer ) ) :neuron  \=  layer \[ j \]neuron \[ 'delta' \]  \=  errors \[ j \]  \*  transfer\_derivative ( neuron \[ 'output' \] )##            neuron\['delta'\] \= errors\[j\] / transfer\_derivative(neuron\['output'\])My understanding is gradient \= dError / dWeights. Therefore, dWeights \= dError / gradient\
 i.e. delta \= errors\[j\] / derivativeDid we somehow make changes here, for calculation reasons, to use arctan instead of tan for gradient?I’d be grateful if you could help.        - 
Dhaila November 22, 2020 at 12:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-577359 "Direct link to this comment")Hi Dong,I was looking into the code. And have the same ques as you raised above. That why we are multiplying. Can I please ask you if you get any understanding of that?            - 
Francisco December 6, 2022 at 10:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-687791 "Direct link to this comment")Hi Dhaila, sorry if this comes a bit late, but for anyone wondering why it is multiplied and not divided, it is due to the chain rule. The core idea of backpropagation is to find the gradient of the cost function i.e. error with respect to the weights, in other words, dE/dw. However, the error we have computed is (label-output), which is equivalent to dE/dy; then, we have computed the derivative from the neuron, which is dy/dw. Hence, by multiplying, you will get dE/dy \*dy/dw \= dE/dw which is what we are looking for. This explanation is simplified, if you would like a more in-depth answer, I would suggest reading chapter 8 from Deep Learning by Ian Goodfellow or Machine learning by Bishop. They go into more depth about this topic. Also, Jason, feel free to correct me if you think I might have misrepresented anything        - 
fernando May 5, 2024 at 1:16 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-714402 "Direct link to this comment")it is not a division. it is a derivative operator. d/dWeights applied on Error.    - 
Maria January 12, 2020 at 5:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-517755 "Direct link to this comment")Hi Jason ,I need code of back propagation artificial neural network for predicting population dynamics of insects pests.        - 
[Jason Brownlee](https://machinelearningmastery.com) January 13, 2020 at 8:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-517804 "Direct link to this comment")Sounds like a great project. Perhaps start here:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)2. 
MO November 8, 2016 at 9:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-369860 "Direct link to this comment")where can i see your data set, i want to see how it looked like    - 
[Jason Brownlee](https://machinelearningmastery.com) November 8, 2016 at 10:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-369876 "Direct link to this comment")Hi MO.The small contrived dataset used for testing is listed inline in the post in section 4.2The dataset used for the full example is on the UCI ML repository, linked in the section titled “Wheat Seeds Dataset”. Here is the direct link:\
 [http://archive.ics.uci.edu/ml/datasets/seeds](http://archive.ics.uci.edu/ml/datasets/seeds)        - 
Solene EBA March 4, 2022 at 11:56 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-658072 "Direct link to this comment")Hello do you have any ideas to calculate the Rsquared            - 
James Carmichael March 5, 2022 at 12:36 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-658136 "Direct link to this comment")Hi Solene..Please clarify what code listing you have a question about so that I may better assist you.3. 
prakash November 11, 2016 at 12:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370232 "Direct link to this comment")in two class classification for 0 the expected value is \[1,0\] for 1 its is \[0,1\].\
 how will be the output vectors for more than two class??    - 
[Jason Brownlee](https://machinelearningmastery.com) November 11, 2016 at 10:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370268 "Direct link to this comment")Hi prakash,For multi-class classification, we can extend the one hot encoding.Three class values for “red”, “green” “blue” can be represented as an output vector like:\
 1, 0, 0 for red\
 0, 1, 0 for green\
 0, 0, 1 for blueI hope that helps.4. 
Rakesh November 13, 2016 at 3:41 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370436 "Direct link to this comment")Hi, Jason.\
 You’ve mentioned that there are 3 output classes.\
 How do we check the values which come under the 3 classes / clusters?\
 Could we print the data which fall under each class?    - 
[Jason Brownlee](https://machinelearningmastery.com) November 14, 2016 at 7:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370498 "Direct link to this comment")Hi Rakesh,The data does belong to 3 classes. We can check the skill of our model by comparing the predicted classes to the actual/expected classes and calculate an accuracy measure.5. 
Alex November 16, 2016 at 12:35 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370823 "Direct link to this comment")I’m confused why the activation method iterates from 0 to len(inputs) – 1 instead of from 0 to len(weights) – 1. Am I missing something?    - 
[Jason Brownlee](https://machinelearningmastery.com) November 17, 2016 at 9:47 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370925 "Direct link to this comment")Hi Alex,The length of weights is the length of the input \+ 1 (to accommodate the bias term).We add the bias term first, then we add the weighted inputs. This is why we iterate over input values.Does that help?        - 
Alex November 17, 2016 at 12:29 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-370953 "Direct link to this comment")When I step through the code above for the ‘forward\_propagate’ test case, I see the code correctly generate the output for the single hidden node but that output doesn’t get correctly processed when determining the outputs for the output layer. As written above in the activate function ‘for i in range(len(inputs)-1):’, when the calculation gets to the activate function for the output node for class\=0, since ‘inputs’ has a single element in it (the output from the single hidden node), ‘len(inputs) – 1’ equals 0 so the for loop never executes. I’m assuming the code is supposed to read ‘for i in range(len(weights) -1):’ Does that make sense?I’m just trying to make sure I don’t fundamentally misunderstand something and improve this post for other readers. This site has been really, really helpful for me.            - 
[Jason Brownlee](https://machinelearningmastery.com) November 18, 2016 at 8:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-371044 "Direct link to this comment")I’m with you now, thanks for helping me catch-up. Nice spot. I’ll fix up the tutorial.Update: Fixed. Thanks again mate!6. 
Tomasz Panek November 21, 2016 at 1:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-371350 "Direct link to this comment")# Update network weights with error\
 def update\_weights(network, row, l\_rate):\
 for i in range(len(network)):\
 inputs \= row\
 if i !\= 0:\
 inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]\
 for neuron in network\[i\]:\
 for j in range(len(inputs)-1):\
 neuron\[‘weights’\]\[j\] \+\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] \+\= l\_rate \* neuron\[‘delta’\]In this fragment:\
 for j in range(len(inputs)-1):\
 neuron\[‘weights’\]\[j\] \+\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] \+\= l\_rate \* neuron\[‘delta’\]If inputs length \= 1, you are not updating weights, it’s correct? You are updating only bias, because in hidden layer is only one neuron.7. 
Tomasz November 21, 2016 at 1:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-371352 "Direct link to this comment")Hello. In method update\_weight you are doing for j in range(len(inputs) – 1). If inputs lenght \= 1, you aren’t updating weights. It’s correct? Hidden layer have one neuron so in output layer weights aren’t updated    - 
[Jason Brownlee](https://machinelearningmastery.com) November 22, 2016 at 6:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-371489 "Direct link to this comment")Hi Tomasz, The assumption here is that the input vector always contains at least one input value and an output value, even if the output is set to None.You may have found a bug though when updating the layers. I’ll investigate and get back to you.        - 
[Jason Brownlee](https://machinelearningmastery.com) January 3, 2017 at 10:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-379663 "Direct link to this comment")Thanks Tomasz, this was indeed a bug.I have updated the update\_weights() function in the above code examples.            - 
Jerry Jones October 16, 2018 at 8:18 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451747 "Direct link to this comment")I don’t understand how update\_weights updates the NN. There is no global variable or return from the function. What am I missing?                - 
[Jason Brownlee](https://machinelearningmastery.com) October 16, 2018 at 2:33 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451765 "Direct link to this comment")The weights are passed in by reference and modified in place.This is an advanced tutorial, I’d recommend using Keras for beginners.8. 
Michael December 13, 2016 at 4:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-374990 "Direct link to this comment")Hi, Thanks for the tutorial, I’m doing a backpropagation project at the moment so its been really useful. I was a little confused on the back-propagation error calculation function. Does “if i !\= len(network)-1:” mean that if the current layer isn’t the output layer then this following code is run or does it mean that the current layer is an output layer?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 13, 2016 at 8:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-375013 "Direct link to this comment")Glad to hear it Michael.The line means if the index i is not equal to the index of the last layer of the network (the output layer), then run code inside the block.9. 
Michael January 5, 2017 at 7:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-380227 "Direct link to this comment")I have another question.\
 Would it be possible to extend the code from this tutorial and create a network that trains using the MNIST handwritten digit set? using a input unit to represent each pixel in the image. I’m also not sure whether/how I could use feature extractors for the images.I have a project where I have to implement the Backpropagation algorithm with possibly the MNIST handwritten digit training set. I hope my question makes sense!    - 
[Jason Brownlee](https://machinelearningmastery.com) January 5, 2017 at 9:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-380268 "Direct link to this comment")Sure Michael, but I would recommend using a library like Keras instead as this code is not written for performance.Load an image as a long list of pixel integer values, convert to floats and away you go. No feature extraction needed for a simple MLP implementation. You should get performance above 90%.10. 
Calin January 6, 2017 at 10:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-380650 "Direct link to this comment")Hi Jason,Great post!I have a concern though:In train\_network method there are these two lines of code:expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1Couldn’t be the case that expected\[row\[-1\]\] \= 1 will throw IndexError, as n\_outputs is the size of the training set which is a subset of the dataset and row basically contains values from the whole dataset?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 7, 2017 at 8:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-380745 "Direct link to this comment")Hi Calin,If I understand you correctly, No. The n\_outputs var is the length of the number of possible output values.Maybe put some print() statements in to help you better understand what values variables have.        - 
Calin January 7, 2017 at 9:48 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-380871 "Direct link to this comment")Hmm..I ran the entire code (with the csv file downloaded from [http://archive.ics.uci.edu/ml/datasets/seeds](http://archive.ics.uci.edu/ml/datasets/seeds)), added some breakpoints and this is what I got after a few iterations:n\_outputs \= 168\
 row\[-1\] \= 201which is causing IndexError: list assignment index out of range.            - 
Adriaan January 11, 2017 at 4:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381390 "Direct link to this comment")I’ve got the same error, That my list assignment index is out of range                - 
[Jason Brownlee](https://machinelearningmastery.com) January 11, 2017 at 9:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381425 "Direct link to this comment")Sorry to hear that, did you try running the updated code?                - 
Ivan January 16, 2017 at 10:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-382373 "Direct link to this comment")This is error of csv read. Try to reformat it with commas. For me it worked                - 
[Jason Brownlee](https://machinelearningmastery.com) January 16, 2017 at 10:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-382386 "Direct link to this comment")What was the problem and fix exactly Ivan?                - 
Bob February 5, 2017 at 10:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-386073 "Direct link to this comment")The data file ([http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds\_dataset.txt](http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt)) has a few lines with double tabs (\t\t) as the delimiter — removing the double tabs and changing tabs to commas fixed it.Thanks for the good article.                - 
[Jason Brownlee](https://machinelearningmastery.com) February 6, 2017 at 9:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-386187 "Direct link to this comment")Thanks for the note Bob.                - 
[Rowen Bruce](http://www.hit.ac.ze) October 20, 2018 at 8:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-452204 "Direct link to this comment")updated code    - 
Adriaan January 11, 2017 at 5:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381397 "Direct link to this comment")I’ve had the same error at the ‘train\_network’ function. Is your dataset fine? I’ve had some problems because the CSV file wasn’t loaded correctly due to my regional windows settings. I’ve had to adjust my settings and everything worked out alright. [http://superuser.com/questions/783060/excel-save-as-csv-options-possible-to-change-comma-to-pipe-or-tab-instead](http://superuser.com/questions/783060/excel-save-as-csv-options-possible-to-change-comma-to-pipe-or-tab-instead)11. 
Stanley January 8, 2017 at 3:15 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381009 "Direct link to this comment")Thanks for such a good article. Just one question: in the equation “weight \= weight \+ learning\_rate \* error \* input”, why there is an “input”? IMO it should be: “weight \= weight \+ learning\_rate \* error”?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 9, 2017 at 7:47 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381130 "Direct link to this comment")The var names and explanation are correct.The update equation is:1weight  \=  weight  \+  learning\_rate \*  error \*  inputFor the input layer the input are the input data, for hidden layers the input is the output of the prior layer.        - 
Herman October 21, 2021 at 6:33 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-630965 "Direct link to this comment")I think the formula should be weight \= weight – learning\_rate \* error \* input instead of \+. Am I right?            - 
Adrian Tam October 22, 2021 at 3:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-631085 "Direct link to this comment")You’re right if you comparing what it is done here to your textbook! However, notice the line “errors.append(expected\[j\] – neuron\[‘output’\])”, hence the error is expressed negative of what you expect. So this is corrected.Probably I should revise the code to make it consistent with other people’s implementation.12. 
Madwadasa January 13, 2017 at 3:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381762 "Direct link to this comment")Jason, Thanks for the code and post.\
 Why is “expected” in expected \= \[0 for i in range(n\_outputs)\] initialized to \[0,0\] ?\
 Should not the o/p values be taken as expected when training the model ?\
 i.e for example in case of Xor should not 1 be taken as the expected ?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 13, 2017 at 9:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-381810 "Direct link to this comment")Hi Madwadasa,Expected is a one-hot encoding. All classes are “0” expect the actual class for the row which is marked as a “1” on the next line.13. 
Michael January 19, 2017 at 3:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-382858 "Direct link to this comment")Hello, I have a couple more questions. When training the network with a dataset, does the error at each epoch indicate the distance between the predicted outcomes and the expected outcomes together for the whole dataset? Also when the mean accuracy is given in my case being 13% when I used the MNIST digit set, does this mean that the network will be correct 13% of the time and would have an error rate of 87%?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 19, 2017 at 7:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-382900 "Direct link to this comment")Hi Michael,The epoch error does capture how wrong the algorithm is on all training data. This may or may not be a distance depending on the error measure used. RMSE is technically not a distance measure, you could use Euclidean distance if you like, but I would not recommend it.Yes, in generally when the model makes predictions your understanding is correct.14. 
Bernardo Galvão January 24, 2017 at 3:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-383717 "Direct link to this comment")Hi Jason,in the excerpt regarding error of a neuron in a hidden layer:“Where error\_j is the error signal from the jth neuron in the output layer, weight\_k is the weight that connects the kth neuron to the current neuron and output is the output for the current neuron.”is the k-th neuron a neuron in the output layer or a neuron in the hidden layer we’re “on”? What about the current neuron, are you referring to the neuron in the output layer? Sorry, english is not my native tongue.Appreciate your work!Bernardo15. 
anonymous February 1, 2017 at 1:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-385291 "Direct link to this comment")It would have been better if recall and precision were printed. Can somebody tell me how to print them in the above code.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 1, 2017 at 10:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-385423 "Direct link to this comment")You can learn more about precision and recall here:\
 [https://en.wikipedia.org/wiki/Precision\_and\_recall](https://en.wikipedia.org/wiki/Precision_and_recall)16. 
kehinde kolade February 6, 2017 at 8:29 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-386259 "Direct link to this comment")Hello Jason, great tutorial, I am developer and I do not really know much about this machine learning thing but I need to extend this your code to incorporate the Momentum aspect to the training, can you please explain how I can achieve this extension?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 7, 2017 at 10:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-386365 "Direct link to this comment")Sorry, I don’t have the capacity to write or spell out this change for you.My advice would be to read a good book on the topic, such as Neural Smithing: [http://amzn.to/2ld9ds0](http://amzn.to/2ld9ds0)17. 
ibrahim February 18, 2017 at 2:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-388719 "Direct link to this comment")Hi Jason,\
 I have my own code written in C\+\+, which works similar to your code. My intention is to extend my code to convolutional deep neural nets, and i have actually written the convolution, Relu and pooling functions however i could not begin to apply the backpropagation i have used in my shallow neural net, to the convolutional deep net, cause i really cant imagine the transition of the backpropagation calculation between the convolutional layers and the standard shallow layers existing in the same system. I hoped to find a source for this issue however i always come to the point that there is a standard backpropagation algorithm given for shallow nets that i applied already. Can you please guide me on this problem?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 18, 2017 at 8:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-388796 "Direct link to this comment")I”d love to guide you but I don’t have my own from scratch implementation of CNNs, sorry. I’m not best placed to help at the moment.I’d recommend reading code from existing open source implementations.Good luck with your project.18. 
[matias](http://cingkleung.com) February 22, 2017 at 3:34 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-389517 "Direct link to this comment")Thank you, I was looking for exactly this kind of ann algorith. A simple thank won’t be enough tho lol    - 
[Jason Brownlee](https://machinelearningmastery.com) February 23, 2017 at 8:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-389639 "Direct link to this comment")I’m glad it helped.The best way to help is to share the post with other people, or maybe purchase one of my books to support my ongoing work:\
 [https://machinelearningmastery.com/products](https://machinelearningmastery.com/products)19. 
Manohar Katam February 26, 2017 at 3:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390209 "Direct link to this comment")Great one! .. I have one doubt .. the dataset seeds contains missing features/fields for some rows.. how you are handling that …    - 
[Jason Brownlee](https://machinelearningmastery.com) February 27, 2017 at 5:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390313 "Direct link to this comment")You could set the missing values to 0, you could remove the rows with missing values, you could impute the missing values with mean column values, etc.Try a few different methods and see what results in the best performing models.        - 
Manohar Katam March 1, 2017 at 2:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390756 "Direct link to this comment")What if I have canonical forms like “male” or “female” in my dataset… Will this program work even with string data..            - 
[Jason Brownlee](https://machinelearningmastery.com) March 2, 2017 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390866 "Direct link to this comment")Hi Manohar,No, you will need to convert them to integers (integer encoding) or similar.20. 
Wissal ARGOUBI February 27, 2017 at 11:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390434 "Direct link to this comment")Great job! this is what i was looking for ! thank you very much .\
 However i already have a data base and i didn’t know how to make it work with this code how can i adapt it on my data\
 Thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) February 28, 2017 at 8:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-390485 "Direct link to this comment")This process will help you work through your predictive modeling problem:\
 [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)21. 
Shweta Gupta March 5, 2017 at 4:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-391271 "Direct link to this comment")Thanks for such a great article..\
 I have one question, in update\_weights why you have used weight\=weight\+l\_rate\*delta\*input rather than weight\=weight\+l\_rate\*delta?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 6, 2017 at 10:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-391461 "Direct link to this comment")You can learn more about the math in the book on the topic. I recommend Neural Smithing: [http://amzn.to/2ld9ds0](http://amzn.to/2ld9ds0)22. 
Sittha March 13, 2017 at 1:23 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-392458 "Direct link to this comment")Thanks for a good tutorial.\
 I have some IndexError: list assignment index out of range. And I cannot fix it with comma or full-stop separator.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 14, 2017 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-392571 "Direct link to this comment")What is the full error you are getting?Did you copy-paste the full final example and run it on the same dataset?        - 
Sittha March 24, 2017 at 3:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-393944 "Direct link to this comment")line 151 :\
 expected\[row\[-1\]\] \= 1\
 IndexError : list assignment index out of range            - 
[Jason Brownlee](https://machinelearningmastery.com) March 24, 2017 at 8:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-393980 "Direct link to this comment")Is this with a different dataset?                - 
Benji Weiss May 11, 2017 at 5:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399327 "Direct link to this comment")if it is a different dataset, what do i need to do to not get this error23. 
Karan March 16, 2017 at 6:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-392903 "Direct link to this comment")The dataset that was given was for training the network. Now how do we test the network by providing the 7 features without giving the class label(1,2 or 3) ?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 17, 2017 at 8:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-392984 "Direct link to this comment")You will have to adapt the example to fit the model on all of the training data, then you can call predict() to make predictions on new data.        - 
Karan March 19, 2017 at 7:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-393320 "Direct link to this comment")Ok Jason, i’ll try that and get back to you! Thank you!24. 
Karan March 19, 2017 at 7:48 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-393321 "Direct link to this comment")Just a suggestion for the people who would be using their own dataset(not the seeds\_dataset) for training their network, make sure you add an IF loop as follows before the 45th line :\
 if minmax\[i\]\[1\]!\=minmax\[i\]\[0\]This is because your own dataset might contain same values in the same column and that might cause a divide by zero error.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 20, 2017 at 8:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-393371 "Direct link to this comment")Thanks for the tip Karan.25. 
[Li Qun](https://liquntang.wordpress.com/) March 25, 2017 at 5:45 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394130 "Direct link to this comment")Thanks jason for the amazing posts of your from scratch pyhton implementations! i have learned so much from you!I have followed through both your naive bayes and backprop posts, and I have a (perhaps quite naive) question: what is the relationship between the two? did backprop actually implement bayesian inference (after all, what i understand is that bayesian \= weights being updated every cycle) already? perhaps just non-gaussian? so.. are non-gaussian PDF weight updates not bayesian inference?i guess to put it simply : is backpropagation essentially a bayesian inference loop for an n number of epochs?I came from the naive bayes tutorial wanting to implement backpropagation together with your naive bayes implementation but got a bit lost along the way.sorry if i was going around in circles, i sincerely hope someone would be able to at least point me on the right direction.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 26, 2017 at 6:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394185 "Direct link to this comment")Great question.No, they are both very different. Naive bayes is a direct use of the probabilities and bayes theorem. The neural net is approximating a mapping function from inputs and outputs – a very different approach that does not directly use the joint probability.26. 
Chiraag March 26, 2017 at 10:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394245 "Direct link to this comment")How did you decide that the number of folds will be 5 ? Could you please explain the significance of this number. Thank You.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 27, 2017 at 7:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394290 "Direct link to this comment")In this case, it was pretty arbitary.Generally, you want to split the data so that each fold is representative of the dataset. The objective measure is how closely the mean performance reflect the actual performance of the model on unseen data. We can only estimate this in practice (standard error?).27. 
[Li Qun](https://liquntang.wordpress.com/) March 27, 2017 at 10:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394355 "Direct link to this comment")Dear Jason,thank you for the reply! I read up a bit more about the differences between Naive Bayes (or Bayesian Nets in general) and Neural Networks and found this Quora answer that i thought was very clear. I’ll put it up here to give other readers a good point to go from: [https://www.quora.com/What-is-the-difference-between-a-Bayesian-network-and-an-artificial-neural-network](https://www.quora.com/What-is-the-difference-between-a-Bayesian-network-and-an-artificial-neural-network)TL:DR :\
 – they look the same, but every node in a Bayesian Network has meaning, in that you can read a Bayesian network structure (like a mind map) and see what’s happening where and why.\
 – a Neural Network structure doesn’t have explicit meaning, its just dots that link previous dots.\
 – there are more reasons, but the above two highlighted the biggest difference.Just a quick guess after playing around with backpropagation a little: the way NB and backprop NN would work together is by running Naive Bayes to get a good ‘first guess’ of initial weights that are then run through and Neural Network and Backpropagated?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 28, 2017 at 8:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394407 "Direct link to this comment")Please note that a Bayesian network and naive bayes are very different algorithms.28. 
Melissa March 27, 2017 at 10:54 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394359 "Direct link to this comment")Hi Jason,\
 Further to this update: Update Jan/2017: Changed the calculation of fold\_size in cross\_validation\_split() to always be an integer. Fixes issues with Python 3.I’m still having this same problem whilst using python 3, on both the seeds data set and my own. It returns an error at line 75 saying ‘list object has no attribute ‘sum” and also saying than ‘an integer is required.’Any help would be very much appreciated.\
 Overall this code is very helpful. Thank you!    - 
[Jason Brownlee](https://machinelearningmastery.com) March 28, 2017 at 8:24 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394408 "Direct link to this comment")Sorry to hear that, did you try copy-paste the complete working example from the end of the post and run it on the same dataset from the command line?        - 
Melissa March 28, 2017 at 9:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394418 "Direct link to this comment")Yes I’ve done that, but still the same problem!29. 
david March 29, 2017 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-394538 "Direct link to this comment")Hello jason,please i need help on how to pass the output of the trained network into a fuzzy logic system if possible a code or link which can help understand better. Thank you30. 
Aditya April 2, 2017 at 3:57 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395134 "Direct link to this comment")Awesome Explanation    - 
[Jason Brownlee](https://machinelearningmastery.com) April 4, 2017 at 9:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395321 "Direct link to this comment")Thanks!31. 
Raunak Jain April 6, 2017 at 5:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395516 "Direct link to this comment")Hello Jason\
 I m getting list assignment index out or range error. How to handle this error?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 9, 2017 at 2:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395789 "Direct link to this comment")The example was developed for Python 2, perhaps this is Python version issue?32. 
Marco April 6, 2017 at 9:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395538 "Direct link to this comment")Thanks but I think python is not a good choice…    - 
[Jason Brownlee](https://machinelearningmastery.com) April 9, 2017 at 2:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395796 "Direct link to this comment")I think it is a good choice for learning how backprop works.What would be a better choice?33. 
Agrawal April 6, 2017 at 9:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395539 "Direct link to this comment")Hey, Jason Thanks for this wonderful lecture on Neural Network.As I am working on Iris Recognition, I have extracted the features of each eye and store it in .csv file, Can u suggest how further can I build my Backpropagation code.\
 As when I run your code I am getting many errors.\
 Thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) April 9, 2017 at 2:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395797 "Direct link to this comment")This process will help you work through your modeling problem:\
 [https://machinelearningmastery.com/start-here/#process](https://machinelearningmastery.com/start-here/#process)34. 
Jack April 7, 2017 at 3:42 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395606 "Direct link to this comment")Could you please convert this iterative implementation into matrix implementation?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 9, 2017 at 2:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-395815 "Direct link to this comment")Perhaps in the future Jack.35. 
Jk April 12, 2017 at 5:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396152 "Direct link to this comment")Hi Jason, In section 4.1 , may you please explain why you used ### inputs \= row\[:-1\] ### ?Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) April 12, 2017 at 7:58 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396174 "Direct link to this comment")Yes. By default we are back-propagating the error of the expected output vs the network output (inputs \= row\[:-1\]), but if we are not the output layer, propagate the error from the previous layer in the network (inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]).I hope that helps.        - 
JK April 13, 2017 at 3:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396257 "Direct link to this comment")Thanks for your respond. I understand what you said , the part I am no understanding is the \[:-1\] . why eliminating the last list item ?            - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2017 at 10:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396295 "Direct link to this comment")It is a range from 0 to the second last item in the list, e.g. (0 to n-1)            - 
Amer April 6, 2018 at 7:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-434281 "Direct link to this comment")Because the last Item in the weights array is the biass36. 
Prem Puri April 12, 2017 at 8:18 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396220 "Direct link to this comment")In function call, def backward\_propagate\_error(network, expected):\
 how much i understand is , it sequentially pass upto\
 if i !\= len(network)-1:\
 for j in range(len(layer)):\
 error \= 0.0\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 My question is which value is used in neuron\[‘delta’\]    - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2017 at 10:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396284 "Direct link to this comment")delta is set in the previous code block. It is the error signal that is being propagated backward.        - 
Nishu March 25, 2018 at 11:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-433169 "Direct link to this comment")I’m sorry, but I still can’t find the location where delta is set and hence, the code gives error.\
 Where is the delta set for the first time?37. 
Prem Puri April 14, 2017 at 3:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396366 "Direct link to this comment")Thanks very much!    - 
[Jason Brownlee](https://machinelearningmastery.com) April 14, 2017 at 8:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-396395 "Direct link to this comment")You’re welcome.38. 
youssef oumate April 26, 2017 at 4:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-397669 "Direct link to this comment")Hi Jason Thank you very much for this awesome implementation of neural network,\
 I have a question for you : I want to replace the activation function from Sigmoid\
 to RELU . So, what are the changes that I should perform in order to get\
 correct predictions?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 27, 2017 at 8:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-397730 "Direct link to this comment")I think just a change to the transfer() and transfer\_derivative() functions will do the trick.        - 
youssef oumate April 27, 2017 at 10:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-397752 "Direct link to this comment")Awesome !Thank you so much            - 
[Jason Brownlee](https://machinelearningmastery.com) April 28, 2017 at 7:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-397833 "Direct link to this comment")You’re welcome.        - 
audrey April 14, 2020 at 5:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529638 "Direct link to this comment")how? please            - 
[Jason Brownlee](https://machinelearningmastery.com) April 14, 2020 at 6:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529674 "Direct link to this comment")If you need help coding relu, see this:\
 [https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)39. 
Yahya Alaa April 30, 2017 at 2:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398019 "Direct link to this comment")Hi Jason,\
 Thank you very much for this wonderful implementation of Neural Network, it really helped me a lot to understand neural networks concept,n\_inputs \= len(dataset\[0\]) – 1\
 n\_outputs \= len(set(\[row\[-1\] for row in dataset\]))\
 network \= initialize\_network(n\_inputs, 2, n\_outputs)\
 train\_network(network, dataset, 0.5, 20, n\_outputs)What do n\_inputs and n\_outputs refer to? According to the small dataset used in this section, is n\_inputs only 2 and n\_outputs only 2 (0 or 1) or I am missing something?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 30, 2017 at 5:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398039 "Direct link to this comment")Input/outputs refers to the number of input and output features (columns) in your data.    - 
Yahya Alaa May 3, 2017 at 1:42 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398392 "Direct link to this comment")Is the program training the network for 500 epochs for each one of the k-folds and then testing the network with the testing data set?        - 
[Jason Brownlee](https://machinelearningmastery.com) May 4, 2017 at 8:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398484 "Direct link to this comment")Hi Yahya,5-fold cross validation is used.That means that 5 models are fit and evaluated on 5 different hold out sets. Each model is trained for 500 epochs.I hope that makes things clearer Yahya.            - 
Yahya Alaa May 4, 2017 at 8:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398496 "Direct link to this comment")Yes you made things clear to me, Thank you.\
 I have two other questions,\
 How to know when to stop training the network to avoid overfitting?\
 How to choose the number of neurons in the hidden layer?                - 
[Jason Brownlee](https://machinelearningmastery.com) May 5, 2017 at 7:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398571 "Direct link to this comment")You can use early stopping, to save network weights when the skill on a validation set stops improving.The number of neurons can be found through trial and error.                - 
Yahya Alaa May 6, 2017 at 8:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398753 "Direct link to this comment")I am working on a program that recognizes handwritten digits, the dataset is consisting of pictures (45\*45) pixels each, which is 2025 input neurons, this causes me a problem in the activation function, the summation of (weight\[i\] \* input\[i\]) is big, then it gives me always a result of (0.99 -> 1) after putting the value of the activation function in the Sigmoid function, any suggestions?                - 
[Jason Brownlee](https://machinelearningmastery.com) May 7, 2017 at 5:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398864 "Direct link to this comment")I would recommend using a Convolutional Neural Network rather than a Multilayer Perceptron.40. 
morok April 30, 2017 at 3:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398029 "Direct link to this comment")In section 3.2. Error Backpropagation, where did output numbers came from for testing backpropagation‘output’: 0.7105668883115941\
 ‘output’: 0.6213859615555266\
 ‘output’: 0.6573693455986976Perhaps from outputs on test forward propagation \[0.6629970129852887, 0.7253160725279748\] taking dd -> derivative \= output \* (1.0 – output), problem is they don’t match, so I’m a bit lost here…thanks!Awesome article!!!    - 
[Jason Brownlee](https://machinelearningmastery.com) April 30, 2017 at 5:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-398041 "Direct link to this comment")In that example, the output and weights were contrived to test back propagation of error. Note the “delta” in those outputs.    - 
Massa November 25, 2017 at 7:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-421124 "Direct link to this comment")hello Dr Jason…I was wondering …n\_outputs \= len(set(\[row\[-1\] for row in dataset\]))this line, how does it give the number of output features?\
 when I print it gives the number of the dataset(number of rows, not columns)        - 
[Jason Brownlee](https://machinelearningmastery.com) November 25, 2017 at 10:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-421146 "Direct link to this comment")The length of the set of values in the final column.Perhaps this post will help with Python syntax:\
 [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)            - 
Massa November 26, 2017 at 4:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-421189 "Direct link to this comment")but I thought it gives the number of outputs…I mean the number of neurons in the output layer.here it’s giving the number of the dataset ….if I have 200 input/output pairs it prints 200so I am confused…how would it be?                - 
[Jason Brownlee](https://machinelearningmastery.com) November 26, 2017 at 7:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-421205 "Direct link to this comment")If there are two class values, it should print 2. It should not print the number of examples.41. 
Umamaheswaran May 8, 2017 at 9:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399034 "Direct link to this comment")Hi Jason, I am using the MNIST data set to implement a handwritten digit classifier. How many training examples will be needed to get a perfomance above 90%.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 9, 2017 at 7:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399092 "Direct link to this comment")I would recommend using a CNN on MNIST. See this tutorial:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)42. 
Huyen May 9, 2017 at 6:32 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399149 "Direct link to this comment")Hi Jason,Your blog is totally awesome not only by this post but also for the whole series about neural network. Some of them explained so much useful thing than others on Internet. They help me a lot to understand the core of network instead of applying directly Keras or Tensorflow.Just one question, if I would like to change the result from classification to regression, which part in back propagation I need to change and how?Thank you in advance for your answer    - 
[Jason Brownlee](https://machinelearningmastery.com) May 10, 2017 at 8:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399253 "Direct link to this comment")Thanks Huyen.You would change the activation function in the output layer to linear (e.g. no transform).43. 
TGoritsky May 12, 2017 at 12:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399412 "Direct link to this comment")Hi Jason,I am playing around with your code to better understand how the ANN works. Right now I am trying to do predictions with a NN, that is trained on my own dataset, but the program returns me one class label for all rows in a test dataset. I understand, that normalizing dataset should help, but it doesn\`t work (I am using your minmax and normalize\_dataset functions). Also, is there a way to return prediction for one-dimensional dataset?\
 Here is the code (sorry for lack of formatting):\
 def make\_predictions():\
 dataset \= \[\[29,46,107,324,56,44,121,35,1\],\
 \[29,46,109,327,51,37,123,38,1\],\
 \[28,42,107,309,55,32,124,38,1\],\
 \[40,112,287,59,35,121,36,1\],\
 \[27,43,129,306,75,41,107,38,1\],\
 \[28,38,127,289,79,40,109,37,1\],\
 \[29,37,126,292,77,35,100,34,1\],\
 \[30,40,87,48,77,51,272,80,2\],\
 \[26,37,88,47,84,44,250,80,2\],\
 \[29,39,91,47,84,46,247,79,2\],\
 \[28,38,85,45,80,47,249,78,2\],\
 \[28,36,81,43,76,50,337,83,2\],\
 \[28,34,75,41,83,52,344,81,2\],\
 \[30,38,80,46,71,53,347,92,2\],\
 \[28,35,72,45,64,47,360,101,2\]\]\
 network \= \[\[{‘weights’: \[0.09640510259345969, 0.37923370996257266, 0.5476265202749506, 0.9144446394025773, 0.837692750149296, 0.5343300438262426, 0.7679511829130964, 0.5325204151469501, 0.06532276962299033\]}\],\
 \[{‘weights’: \[0.040400453542770665, 0.13301701225112483\]}, {‘weights’: \[0.1665525504275246, 0.5382087395561351\]}, {‘weights’: \[0.26800994395551214, 0.3322334781304659\]}\]\]\
 # minmax \= dataset\_minmax(dataset)\
 # normalize\_dataset(dataset, minmax)\
 for row in dataset:\
 prediction \= predict(network, row)\
 print(‘Expected\=%d, Got\=%d’ % (row\[-1\], prediction))    - 
[Jason Brownlee](https://machinelearningmastery.com) May 12, 2017 at 7:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399443 "Direct link to this comment")I would suggest exploring your problem with the Keras framework:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)44. 
Tomo May 18, 2017 at 6:22 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-399961 "Direct link to this comment")Hi Jason!\
 In the function “backward\_propagate\_error”, when you do this: neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])The derivative should be applied on the activation of that neuron, not to the output . Am I right?? neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(activate(neuron\[‘weights’\], inputs))And inputs is:\
 inputs \= row\[-1\]\
 if i !\= 0:\
 inputs \= \[neuron\[‘output’\] for neuron in self.network\[i-1\]\]Thank you! The post was really helpful!    - 
Adika February 2, 2021 at 2:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-595899 "Direct link to this comment")I think you are right but not sure.45. 
Tina May 26, 2017 at 3:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400540 "Direct link to this comment")Hello Jason!This is a very interesting contribution to the community 🙂\
 Have you tried using the algorithm with other activation functions?\
 I tried with Gaussian, tanh and sinx, but the accuracy was not that high, so I think that I omitted something. What I altered were the activation functions and the derivatives. Is there something else that needs to be changed?    - 
[Jason Brownlee](https://machinelearningmastery.com) June 2, 2017 at 11:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401137 "Direct link to this comment")Sigmoid was the defacto standard for many years because it performs well on many different problems. Now the defacto standard is ReLU.        - 
Manu June 6, 2017 at 8:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401683 "Direct link to this comment")Sigmoid and ReLU are transfer functions right ?\
 Activation function is just the sum of all weights and inputs            - 
[Jason Brownlee](https://machinelearningmastery.com) June 7, 2017 at 7:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401731 "Direct link to this comment")You are correct, but in some frameworks, transfer functions are called activation functions:\
 [https://keras.io/activations/](https://keras.io/activations/)46. 
vishwanathan May 27, 2017 at 8:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400644 "Direct link to this comment")Thanks for the great post. Here is some observation that I am not able to understand. In the back ward propagate you are not taking all the weights and only considering the jth. Can you kindly help understand. I was under the impression that the delta from output is applied across all the weights,\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])    - 
vishwanathan May 27, 2017 at 8:14 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400645 "Direct link to this comment")I understand you do not want to take in the bias weight hence the exclusion of the last weight in neuron. I kind of get stumped on bias.47. 
vishwanathan May 27, 2017 at 9:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400646 "Direct link to this comment")Thanks for the great article. In the backward propagate, the delta value is applied for each weight across the neuron and the error is summed. I am curious why is the delta not applied to individual weights of the neuron and the error summed for that neuron. Can you please clarify?48. 
[Josue](http://jgjgjh) May 29, 2017 at 3:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400706 "Direct link to this comment")Why don’t you split the data into TrainData and TestData, like 80% of the dataset for training and 20% for testing, because if you train with 100% of rows of the dataset and then test some rows of the dataset the accuracy will be good . But if you put new data on the seeds.csv the model will work with less accuracy, Right?    - 
[Jason Brownlee](https://machinelearningmastery.com) June 2, 2017 at 12:16 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401170 "Direct link to this comment")You can, k-fold cross validation generally gives a better estimate of model performance.Once we have the estimate and choose our model, we can fit the final model on all available data and make predictions on new data:\
 [https://machinelearningmastery.com/train-final-machine-learning-model/](https://machinelearningmastery.com/train-final-machine-learning-model/)49. 
[Josue](http://donthave) May 29, 2017 at 11:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400727 "Direct link to this comment")Thanks for the post! I have a question about cross-validation. The dataset of seeds is perfect for 5 folds but for a dataset of 211? I’ll have uniformly sized subset right? (211/5) Can you give me a suggestion how I could handle that ?\
 Thanks in advanced.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 2, 2017 at 12:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401176 "Direct link to this comment")One way is that some records can be discarded to give even sized groups.50. 
Sebastián May 30, 2017 at 9:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-400803 "Direct link to this comment")Thanks so much for the tutorial. It was really helpful!    - 
[Jason Brownlee](https://machinelearningmastery.com) June 2, 2017 at 12:31 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-401194 "Direct link to this comment")I’m glad it helped.51. 
Manu June 10, 2017 at 9:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-402132 "Direct link to this comment")Hello Jason,any advice on how to handle multi-classifier problems when the classes have high cardinality ?\
 I’m thinking about input data of search engines linked to choosen urls.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 11, 2017 at 8:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-402178 "Direct link to this comment")Ouch, consider modeling it as regression instead (e.g. a rating or recommender system).        - 
Manuel June 13, 2017 at 1:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-402304 "Direct link to this comment")Ok thank you very much Jason.\
 But it wont work with searches unseen by the algorithm.\
 I red something in the books “Programming collective intelligence” about a neural net from scratch for this king of problem but I don’t understang how it works for the moments…            - 
[Jason Brownlee](https://machinelearningmastery.com) June 13, 2017 at 8:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-402350 "Direct link to this comment")Consider focusing on one measure/metric that really matters in your domain, then try a suite of framings of the problem and different algorithms to get a feeling for what might work best.52. 
Yash June 18, 2017 at 6:21 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-402977 "Direct link to this comment")I am not able to understand the above code.So, I request you to explain me the above code    - 
[Jason Brownlee](https://machinelearningmastery.com) June 19, 2017 at 8:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-403024 "Direct link to this comment")Which part do you not understand exactly?53. 
Tathagat June 21, 2017 at 3:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-403300 "Direct link to this comment")Hey Jason..am a novice in machine learning..have a small question…how can I track the timesteps involved in the algorithm with accordance with the code?    - 
[Jason Brownlee](https://machinelearningmastery.com) June 22, 2017 at 6:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-403364 "Direct link to this comment")What do you mean by time steps?54. 
bazooka June 29, 2017 at 6:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-404185 "Direct link to this comment")Hi, Jason. I am so confused, in the result, why there are 4 set of \[output,weight,delta\]like this:\
 \[{‘output’: 0.9999930495852168, ‘weights’: \[0.9315463130784808, 1.0639526745114607, 0.9274685127907779\], ‘delta’: -4.508489650980804e-09}, {‘output’: 0.9992087809233077, ‘weights’: \[-2.4595353900551125, 5.153506472345162, -0.5778256160239431\], ‘delta’: 1.940550145482836e-06}\]\
 \[{‘output’: 0.01193860966265472, ‘weights’: \[2.3512725698865053, -8.719060612965613, 1.944330467290268\], ‘delta’: -0.0001408287858584854}, {‘output’: 0.988067899681387, ‘weights’: \[-2.2568526798573116, 8.720113230271012, -2.0392501730513253\], ‘delta’: 0.0001406761850156443}\]after the backpropagation we find the optimal weights to get minimum error, what does these 4 group means?\
 E    - 
[Jason Brownlee](https://machinelearningmastery.com) June 29, 2017 at 7:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-404190 "Direct link to this comment")That is the internal state of the whole trained network.55. 
hassan June 29, 2017 at 7:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-404189 "Direct link to this comment")hi Jason\
 thanks for your code and good description here, i like it so much.\
 i run your example code and encounter with an error same others whom left note here\
 the error is:\
 expected\[row\[-1\]\] \= 1\
 IndexError: list assignment index out of rangehow i can fix this error?    - 
[Jason Brownlee](https://machinelearningmastery.com) June 29, 2017 at 7:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-404193 "Direct link to this comment")The code was written for Python 2.7, confirm that this is your Python version.Also confirm that you have copied the code exactly.56. 
Jerome July 5, 2017 at 9:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-404812 "Direct link to this comment")Dear Jason,i have this question about Back Propagate Error1- derivative sigmoid \= output \* (1.0 – output)\
 That is ok 2- error \= (expected – output) \* transfer\_derivative(output)\
 Ok but it also means that error \=\= 0 for output \= 1 whatever the expected is because transfer\_derivative(1) \=\=0So, whatever the expected , error is nil if output is 1 …\
 Is there something rotten here?ThanksJerome57. 
wddddds July 10, 2017 at 10:01 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-405420 "Direct link to this comment")Thank you Jason, It’s a great tutorial and really helpful for me! But I have to say that trying to reimplement your code strongly increased my ability of debugging 🙂    - 
[Jason Brownlee](https://machinelearningmastery.com) July 11, 2017 at 10:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-405483 "Direct link to this comment")Thanks.58. 
Victor July 17, 2017 at 7:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-406233 "Direct link to this comment")Hi Jason, Thanks for sharing your code. I’m a PhD candidate in machine learning, and I have a doubt about the weights update in section 4.1:weight \= weight \+ learning\_rate \* error \* inputShould not it be as follows?weight \= weight – learning\_rate \* error \* inputThanks again for sharing this.Regards,\
 Victor.    - 
Víctor August 4, 2017 at 11:07 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408545 "Direct link to this comment")I didn’t say anything, my mistake in understanding.Thanks again for sharing your work.59. 
vishnu priya July 22, 2017 at 4:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-406881 "Direct link to this comment")Hi..\
 Thanks for ur coding. It was too helpful. can u suggest me how to use this code for classifying tamil characters. i have tried in cnn and now i need to compare the result with bpn. can u pls suggest me.thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) July 23, 2017 at 6:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-406962 "Direct link to this comment")Perhaps this tutorial on classifying with a CNN would be more useful to you:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)60. 
vishnu priya July 23, 2017 at 4:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407012 "Direct link to this comment")Thank you sir. With this tutorial i have implemented cnn sir. but for BPN i am getting error rate 687.203 sir. i dnt know what to do sir. can u help me sir.Thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) July 24, 2017 at 6:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407079 "Direct link to this comment")What is the problem exactly?61. 
Vishnupriya July 24, 2017 at 4:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407137 "Direct link to this comment")Classification of Tamil characters sir. I have 144 different classes. I have taken 7 glcm features of each character and I need to train this features in backpropagation and predict the character to which class it belongs.    - 
[Jason Brownlee](https://machinelearningmastery.com) July 25, 2017 at 9:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407218 "Direct link to this comment")Sound like a great project!62. 
codeo July 26, 2017 at 5:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407398 "Direct link to this comment")Hi, so I wasn’t following this tutorial when implementing my neural network from scratch, and mine is in JavaScript. I just need help with the theory. How do I calculate the error for each node in the net so that I can incrementally change the weights? Great tutorial btw    - 
codeo July 26, 2017 at 6:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407408 "Direct link to this comment")Hahaha nevermind, it was my code\
 Multidimensional arrays and stuff boggle the mind hah        - 
[Jason Brownlee](https://machinelearningmastery.com) July 27, 2017 at 7:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-407476 "Direct link to this comment")Glad to hear you worked it out.63. 
PRABHAKARAN M July 31, 2017 at 4:31 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408006 "Direct link to this comment")\[ 6.38491205 5.333345 4.81565798 5.43552204 9.96445304 2.57268919 4.07671018 1.5258789 6.19728301 0 1 \]\
 Dear sir,\
 the above mentioned numerical values are extracted from the dental x-ray image using gray level co occurrence matrix \[10 inputs and 1 output\]. This dataset is used as a input for BPN classifier. whether the same data set as\[.csv\] file can be used as the input for DEEP Convolutional Neural Network technique ? and can i get the output as image ? for example if i give the dental x ray images as numerical values i have to get the caries affected teeth as the output for the given dataset.    - 
[Jason Brownlee](https://machinelearningmastery.com) August 1, 2017 at 7:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408079 "Direct link to this comment")That sounds like a great problem. It may be possible.I would recommend using deep CNNs.Perhaps this tutorial will give you some ideas on how to get started:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)You may want to look at some papers on object localization in images. I don’t have material on it sorry.64. 
PRABHAKARAN M July 31, 2017 at 4:32 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408007 "Direct link to this comment")can i get the example code for dental caries detection using deep Convolutional Neural Network for the given dataset as x ray images.    - 
[Jason Brownlee](https://machinelearningmastery.com) August 1, 2017 at 7:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408080 "Direct link to this comment")I do not have sample code for this problem, sorry.65. 
John August 1, 2017 at 3:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408057 "Direct link to this comment")Very nice explanation, thank you.\
 I have some questions.1) weight \= weight \+ learning\_rate \* error \* inputDo I really need to multiply it with input ? For example here [http://home.agh.edu.pl/\~vlsi/AI/backp\_t\_en/backprop.html](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html) they dont it multiply it with input. At least, I think that…2) Is your method same as in [http://home.agh.edu.pl/\~vlsi/AI/backp\_t\_en/backprop.html](http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)?\
 i think yes, but again, Im not sure and Im confused by that input multiplication. 3) What is exactly loss function in your example (I usually found some derivations of loss (cost ?) function (in other explanations), not transfer function derivation)? Im actually very confused by notation which I find around …4) momentum and weight decay. In your example, you can implement them that you substract calculated decay and add calculated momentum (to weight update) ? Again, I found forms which substract both and weight update as w \+ deltaW, so again I’m mega confused by notation for backpropagation which I found…Sorry for dumb questions, … math is not my strong side, so many things which can be inferred by math sense are simply hidden for me.    - 
John August 1, 2017 at 3:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408058 "Direct link to this comment")\*substract both and weight update as w \+ deltaW, so againI found above sentence as nonsense, must be side effect of my confusion …        - 
[Jason Brownlee](https://machinelearningmastery.com) August 1, 2017 at 8:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408096 "Direct link to this comment")Hang in there. Pick one tutorial and focus on it. Jumping from place to place will make things worse for sure.    - 
[Jason Brownlee](https://machinelearningmastery.com) August 1, 2017 at 8:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408094 "Direct link to this comment")Hi John, good questions.According to my textbook, yes.\
 I can’t speak for random sites on the internet sorry.Loss is prediction error. You can change this to other forms like MAE or MSE.No decay or momentum in this example. Easy to add if you want. There are many ways to dial in the learning process. No hard and fast rules, just some norms that people reuse.66. 
Parminder Kaur August 6, 2017 at 7:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408748 "Direct link to this comment")A VERY GOOD TUTORIAL SIR…\
 Sir i am implementing remote sensed image classification using BPN neural network using IDL.\
 I am not finding good resources on constructing features for input dataset and also number of hidden layers and number of neurons in hidden layer.\
 Any resources you know, can help me?Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) August 7, 2017 at 8:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-408800 "Direct link to this comment")The CNN will perform feature extraction automatically, you could explore using different filters on the data to see if it helps the network.The number of layers and neurons/filters per layer must be found using trial and error. It is common to copy the designs from other papers as a starting point.I hope that helps.67. 
pero August 9, 2017 at 1:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-409003 "Direct link to this comment")Nice tutorial, very clean and readable code. \=) thank you!    - 
[Jason Brownlee](https://machinelearningmastery.com) August 9, 2017 at 6:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-409043 "Direct link to this comment")Thanks pero.68. 
Vatandas August 15, 2017 at 3:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-409615 "Direct link to this comment")1. I expect that this code is deep learning (many hidden layer) but not. One sentence is easy (“you can add more hidden layer as explained”) but to do is not as easy as you said.2. I think your code is wrong.\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])\
 but\
 Error \= Target – ActivatedOutputNode\
 Delta \= Error \* Derivative(NONActivatedOutputNode)I mean you use the same ‘output’ variable both error and delta. But in error it must be activated one, in delta it must be NONactivated one.    - 
A Researcher May 2, 2019 at 3:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-483645 "Direct link to this comment")Exactly, this article is completely misleading :S69. 
8CG\_256 August 18, 2017 at 9:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410043 "Direct link to this comment")Nice tutorial, very clean code and beginner-friendly. Thank you very much!    - 
[Jason Brownlee](https://machinelearningmastery.com) August 18, 2017 at 4:36 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410090 "Direct link to this comment")Thanks, I’m glad you found it useful!    - 
8CG\_256 August 18, 2017 at 9:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410125 "Direct link to this comment")I only have one slight issue: I implemented this in Ruby and I tried to train it using the IRIS dataset, keeping the network simple (1 input layer, 1 hidden layer, 1 output layer) and after decreasing for a while the error rate keeps increasing. I tried lowering the learning rate, even making it dynamic so it decreases whenever the error increases but it doesn’t seem to help. Could you give me some advice? P.S sorry for my bad English        - 
[Jason Brownlee](https://machinelearningmastery.com) August 19, 2017 at 6:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410183 "Direct link to this comment")Here is an example of backprop I developed in Ruby:\
 [http://cleveralgorithms.com/nature-inspired/neural/backpropagation.html](http://cleveralgorithms.com/nature-inspired/neural/backpropagation.html)70. 
Derek Martins August 22, 2017 at 9:22 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410592 "Direct link to this comment")Hi Jason, I enjoy so much your tutorials. Can you do a tutorial implementing BackPropagation Through Time? Thanks man.    - 
[Jason Brownlee](https://machinelearningmastery.com) August 23, 2017 at 6:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410658 "Direct link to this comment")Thanks for the suggestion.I have a few posts on the general topic, for example:\
 [https://machinelearningmastery.com/gentle-introduction-backpropagation-time/](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)71. 
[Anubhav Singh](https://xprilion.com) August 24, 2017 at 1:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-410863 "Direct link to this comment")Hello Jason,Thank you for the great tutorial!I would like to know how I can obtain the weight\*input for every single neuron in the network…I’ve been trying these lines – for layer in network:\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])but the activation variable here is a single value…what I understand is that if I have set n\_hidden \= 5 (number of hidden layers), I should get N\*5 (N \= number of features in the dataset) outputs if I print the activation…Kindly help 🙂Thank you!72. 
Jose Panakkel August 25, 2017 at 10:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-411028 "Direct link to this comment")Dear Jason,I have a question on the delta calculation at the output layer, where\
 the primary value is the difference between the neuron output and\
 the expected output. And we are then multiplying this difference\
 with the transfer\_derivative. where transfer\_derivative is a function\
 of neuron’s output.My question is, is it correct to find the difference between the\
 neuron’s output and the expected output?In this case of the example, you have chosen digital outputs \[0,1\]\
 and hence it may not have come up .. but my point is…\
 one is already subjected to a transfer function, and one is not.The neuron’s output is always subjected to a transfer function and\
 hence will be in a specific range, say -.5 to \+.5 or something..\
 But the expected output is the user’s choice .. isnt it?\
 user can have an expected value of say 488.34, for some stock price\
 learning.. then is it still correct to find this primary difference\
 between the expected output and the neuron output, at the output\
 layer delta calculation?shoulnt the expected output also be subjected to the same transfer\
 function before finding the difference? Or the otherway, like\
 shoulnt the neuron ouptut be subjected to a reverse transfer function\
 before comparing with the expected output directly?Thanks and Regards,\
 Jose Panakkel73. 
RealUser404 September 6, 2017 at 1:36 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-412837 "Direct link to this comment")Hello Jason, great tutorial that helped me a lot!I have a question concerning the back-propagation : what if instead of having an error function I only have a desired gradient for the output (in the case of an actor-critic model for example)?\
 How can I change your backprop function to make it work? Or can I just use the gradient as the error?    - 
[Jason Brownlee](https://machinelearningmastery.com) September 7, 2017 at 12:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-412962 "Direct link to this comment")Sorry, I don’t follow, perhaps you can restate your question with an example?74. 
user28 September 8, 2017 at 9:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413123 "Direct link to this comment")Hi Jason , thank you for providing this tutorial. I’m confused of how can I implement the same backpropagation algorithm with output not binary. Since I noticed that your example has binary output. Like predicting for stock price given the open, high, low and close values. Regards.    - 
[Jason Brownlee](https://machinelearningmastery.com) September 9, 2017 at 11:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413225 "Direct link to this comment")Use a library like Keras. Start here:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)75. 
[Lewis](http://na) September 11, 2017 at 2:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413386 "Direct link to this comment")Hi Jason,great article. I have an interest in NN but I am not that good at python. Want I wanted to try was to withhold say 5 rows from the dataset and have the trained network predict the results for those rows. these is is different from what I think the example does which is rolling predictions with the learning. Removing 5 rows from the dataset is of course easy but my pitiful attempts at predicting with unseen data like below fail ((I guess network is not in scope at the end): any help appreciated!# predict unseen data\
 unseendataset \= \[\[12.37,13.47,0.8567,5.204,2.96,3.919,5.001\],\
 \[12.19,13.2,0.8783,5.137,2.981,3.631,4.87\],\
 \[11.23,12.88,0.8511,5.14,2.795,4.325,5.003\],\
 \[13.2,13.66,0.8883,5.236,3.232,8.315,5.056\],\
 \[11.84,13.21,0.8521,5.175,2.836,3.598,5.044\],\
 \[12.3,13.34,0.8684,5.243,2.974,5.637,5.063\]\]for row in unseendataset:\
 prediction2 \= predict(network, row)\
 print(‘Predicted\=%d’ % (prediction2))    - 
[Jason Brownlee](https://machinelearningmastery.com) September 11, 2017 at 12:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413434 "Direct link to this comment")I would recommend starting with Keras rather than coding the algorithm from scratch.Start here:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)76. 
[Karim](https://www.linkedin.com/in/kmagdy/) September 14, 2017 at 1:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413775 "Direct link to this comment")Hi Jason, I am trying to generalize your implementation to work with a variable number of layers and nodes. However, whenever I try to increase the number of nodes too much it stops working (the network freezes at one error rate and all output nodes are active, i.e. giving 1). Although the code would work if I decreased the layers and the errors will go down.\
 Is there something I am missing when using too many layers? The concepts should be the same. I trained a network with 4 layers: \[14,10,10,4\] and it worked.\
 I trained a network with 4 layers \[14,100,40,4\] and it is stuck. Same dataset. My code is here if you are looking in more details:\
 [https://github.com/KariMagdy/Implementing-a-neural-network](https://github.com/KariMagdy/Implementing-a-neural-network)Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) September 15, 2017 at 12:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-413891 "Direct link to this comment")What problem do you get exactly?77. 
Laksh October 4, 2017 at 11:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-415614 "Direct link to this comment")Hi, Jason Brownlee,\
 can we extend this code for 2 or more hidden layers ?    - 
[Jason Brownlee](https://machinelearningmastery.com) October 5, 2017 at 5:24 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-415653 "Direct link to this comment")Sure.78. 
dsliver33 October 9, 2017 at 1:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416085 "Direct link to this comment")Dear Mr. Brownlee,I’m trying to alter the code to represent a regression problem (sigmoid on hidden layer, linear on output layer). As far as I know, the main part of the code that would have to be modified is the FF algorithm. I’ve rewritten the code as below:12345678910111213141516171819# Forward propagate input to a network outputdef forward\_propagate\_regression ( network ,  row ) : inputs \= rownew\_inputs  \=  \[ \] #gets the 1st layer, applies sigmoid activationhiddenlayer  \=  network \[ 0 \]for  neuron in  hiddenlayer :activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )neuron \[ 'output' \]  \=  transfer ( activation )new\_inputs . append ( neuron \[ 'output' \] )inputs  \=  new \_inputs #gets the last layer, applies linear activationoutputlayer  \=  network \[ - 1 \]for  neuron in  outputlayer :activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )neuron \[ 'output' \]  \=  activationnew\_inputs . append ( neuron \[ 'output' \] ) inputs \= new\_inputs return inputsWith this code, I’m getting an “OverflowError: (34, ‘Result too large’)” error. Could you please tell what I’m doing wrong? All the other parts of the code are as you’ve written.    - 
[Jason Brownlee](https://machinelearningmastery.com) October 9, 2017 at 4:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416100 "Direct link to this comment")What did you change exactly? Can you highlight the change for me?Also, try using pre tags.        - 
dsliver33 October 10, 2017 at 4:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416157 "Direct link to this comment")(I don’t know how to highlight the change, sorry!)I got the hidden layer (network\[0\]), and I applied your algorithm (calculate activation, transfer the activation to the output, append that to a new list called “new\_inputs”). After that, I get the output layer (network\[-1\]), I calculate the activation with the “new\_inputs”, but I do NOT apply the sigmoid transfer function (so, the outputs should be linear). The results are appended to a new list, which is set to be the return of the function. Would that be the best way to remove the sigmoid function from the output layer, making the code a regression, instead of a classification?            - 
[Jason Brownlee](https://machinelearningmastery.com) October 10, 2017 at 7:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416186 "Direct link to this comment")Sounds good. I don’t have any good ideas, I’d recommend stepping through some calculations to help spot where it is going wrong.You may want to consider moving to an open source neural net library, such as Keras:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)    - 
Liam McGoldrick October 26, 2017 at 5:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-417956 "Direct link to this comment")I am having the same issue with mine. i made alterations and they are just the same as yours. Did you find a solution?    - 
Liam McGoldrick October 26, 2017 at 5:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-417963 "Direct link to this comment")I GOT IT TO WORK!!! You have to normalize your output data. Then you can apply the transfer function to the output layer just the same! After that it will work!        - 
Steven August 20, 2019 at 8:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497537 "Direct link to this comment")But didn’t you changed the function ‘train\_network’ ???        - 
Urvi Deole March 12, 2021 at 2:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-600592 "Direct link to this comment")Could you please mention the functions you made changes to to get the code to work for regression?79. 
Chris October 12, 2017 at 11:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416406 "Direct link to this comment")Hi Jason, nice posting and it really helps a lot\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])\
 Should the neuron\[‘output’\] be the output of the activation function instead of the transfer function here?80. 
Asad October 14, 2017 at 3:24 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416652 "Direct link to this comment")hi jason, nice post its really helps alot.please tell me how we can change the neuron in hidden layer and in output layer?\
 and what will be the result when we change the neuron in hidden layer and in output layer?\
 in this tutorial u take one hidden layer,so can we use more than one hidden layer? and how?please tell me i m waiting    - 
[Jason Brownlee](https://machinelearningmastery.com) October 15, 2017 at 5:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416721 "Direct link to this comment")Perhaps you would be better served by starting with a neural network library such as Keras:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)81. 
dsliver33 October 16, 2017 at 2:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416872 "Direct link to this comment")Dear Mr. Brownlee,I’m trying to adapt the code to support many hidden layers. I’ve adapted the code as below, with a new input called “n\_layers”, to insert N hidden layers in the network. When I try to run the code, it shows the error below. Do you have any idea why? in backward\_propagate\_error(network, expected)\
 78 error \= 0.0\
 79 for neuron in network\[i \+ 1\]:\
 —> 80 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 81 errors.append(error)\
 82 else:IndexError: list index out of range    - 
dna\_remaps February 3, 2018 at 10:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-428308 "Direct link to this comment")This took me a minute to figure out myself.You need to add a conditional after your first layer to make sure your subsequent hidden layer weights have the proper dimensions (n\_hidden\+1, n\_hidden)for i in range(n\_layers):\
 hidden\_layer \= \[{‘weights’:\[random() for i in range(n\_inputs \+ 1)\]} for i in range(n\_hidden)\]\
 if i > 0:\
 hidden\_layer \= \[\[{‘weights’:\[random() for i in range(n\_hidden \+ 1)\]} for i in range(n\_hidden)\]\
 network.append(hidden\_layer)82. 
Arijit Mukherjee October 17, 2017 at 1:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-416944 "Direct link to this comment")Hi, In the output/last layer when we are calculating the backprop error why are we multiplying with the transfer derivative with the (expected-output)?? transfer derivative is already canceled out for the the last layer , the update should be only (expected-output)\*previous\_layer\_input , ???\
 Thanks83. 
Tanoh Henry October 18, 2017 at 8:54 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-417112 "Direct link to this comment")Really good article. Thanks a lot.\
 Need a little bit of clarification.\
 For backward propagation starting at the output layer,\
 you get the error by appending to errors expected\[j\] – neuron\[‘output’\].\
 Isn’t Error \= 0.5 \* sum(errors)?\
 and then using this sum of errors for back-propagation?\
 Thanks.84. 
Liam October 21, 2017 at 5:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-417341 "Direct link to this comment")Thanks for the tutorial! I am trying to modify your code to do a regression model and I am stuck. I have an input data set (4 columns and many rows) and a single variable output data set (in range of tens of thousands). I fed them into the train procedure and I get an error when it reaches “expected \= \[0 for i in range(n\_outputs)\]” in the train portion. The error reads “only length-1 arrays can be converted to Python scalar”. Now I understand this is because of the intended purpose for the code was a categorization problem but I am wondering what I would need to modify to get this to work? Any help would go a long way as I have been stuck on this issue for some time now.Thanks, and again wonderful tutorial!    - 
[Jason Brownlee](https://machinelearningmastery.com) October 21, 2017 at 5:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-417348 "Direct link to this comment")Perhaps start with Keras, it will be much easier for you:\
 [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)85. 
Sam October 26, 2017 at 11:15 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-418032 "Direct link to this comment")Hi\
 I am implementing a 2 layer neural network with 100 hidden units in the first layer and 50\
 in the next using your code. Implement sigmoid activation function in each layer. Train/test your\
 model on the MNIST dataset subset.\
 But it is always giving same prediction.\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=1, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=1, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=1, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=1, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=1, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1\
 \[0.99999999986772, 0.99999999994584\]\
 Expected\=0, Got\=1    - 
[Jason Brownlee](https://machinelearningmastery.com) October 27, 2017 at 5:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-418053 "Direct link to this comment")I would recommend using a framework like Keras:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)    - 
Matthias April 15, 2018 at 2:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435029 "Direct link to this comment")Weights should be initialized with normally distributed random values. Try using random.gauss for weight initialization.86. 
John October 28, 2017 at 6:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-418240 "Direct link to this comment")help, I dont know why i got this error.Traceback (most recent call last):\
 File “a.py”, line 185, in\
 for i in range(len(dataset\[0\])-1):\
 TypeError: ‘NoneType’ object has no attribute ‘\_\_getitem\_\_’    - 
[Jason Brownlee](https://machinelearningmastery.com) October 29, 2017 at 5:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-418288 "Direct link to this comment")You cannot have the “-1” within the call to len()87. 
João Costa November 12, 2017 at 12:46 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-419811 "Direct link to this comment")Hey Jason, thanks for your post!This is helping me a lot with a college work. But in this NN, how can I set manually not the number of input neuros, the input values? For example, if I have 1 input neuro, I wan’t to set this value to 0.485.Best regards!    - 
[Jason Brownlee](https://machinelearningmastery.com) November 13, 2017 at 10:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-419867 "Direct link to this comment")Sorry, I don’t follow.Perhaps you’d be better off using a library like Keras:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)88. 
yesta November 17, 2017 at 8:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-420277 "Direct link to this comment")Hi, Jason\
 Thank you for this amazing tutorial!I have a question that may be out of the topic. How do you call models or type of DL models where you feed a model with new test data in order to make the model adaptive to the environment?Thank you.    - 
[Jason Brownlee](https://machinelearningmastery.com) November 17, 2017 at 9:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-420299 "Direct link to this comment")Yes, you can update a model after it has been trained.89. 
Nil December 6, 2017 at 7:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422285 "Direct link to this comment")Hi, Dr. Jason,I have been studying how to develop a neural network from scratch and this tutorial is the main one I have been following because it is helping me so much.\
 I have a doubt: When I study the theory I see the neural network scheme carrying only the weights and bias. And here in practice I see that the network is also carrying the output values and the delta i.e (weights, bias, output and delta). Will the final model be saved like this? with the latter (weights, bias, output and delta)? would this be the rule in practice?I would appreciate it if you could help with this issue so that I could get out of where I left off.Your posts are really very good there is where I find my way in to learning in Machine Learning.Best Regards    - 
[Jason Brownlee](https://machinelearningmastery.com) December 7, 2017 at 7:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422347 "Direct link to this comment")The final model (e.g. trained) only needs to perform the forward pass.        - 
Nil December 8, 2017 at 5:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422467 "Direct link to this comment")Understood.\
 Thank you Dr. Jason90. 
MohamedElshazly December 8, 2017 at 9:30 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422514 "Direct link to this comment")Hi , there’s something i don’t understand : 1234567for  i  in  reversed ( range ( len ( network ) ) ) :layer  \=  network \[ i \]errors  \=  list ( )if  i  !\=  len ( network ) - 1 :for  j  in  range ( len ( layer ) ) : error \= 0.0for  neuron in  network \[ i  \+  1 \] :wouldn’t the last line be out of range because the current ‘ i ‘ is the last one and i can’t go beyond it by 1 ? thanks in advance    - 
[Jason Brownlee](https://machinelearningmastery.com) December 9, 2017 at 5:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422538 "Direct link to this comment")No, because of the “if” check on the 4th line down.91. 
Olu December 9, 2017 at 3:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422524 "Direct link to this comment")Hi Mr Brownlee,Thank you for your tutorial. The training for the example worked however when I try to implement the code for the Wheat Seeds Dataset I get an error from my line 210:for i in range(len(dataset\[0\]) – 1):\
 str\_column\_to\_float(dataset, i)The error is: IndexError: list index out of rangeCan you please explain why it is (dataset\[0\])? Does (dataset\[0\]) means the 1st column in the dataset?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 9, 2017 at 5:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422544 "Direct link to this comment")Yes, I recommend learning more about Python arrays here:\
 [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)The example was written for Python 2.7, confirm your Python version.92. 
[Jonesy](http://optimistic.ninja) December 12, 2017 at 3:17 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422862 "Direct link to this comment")Hello Jason,Fantastic stuff here. I had a question about the network configuration. This 2 input, 2 hidden and 2 output seems a bit odd to me. I’m used to seeing 2, 2, 1 for XOR – can you explain why you have two output nodes and how they work with the max function? I think it would better explain this line for me in train():expected\[row\[-1\]\] \= 1And lastly, why would one choose this configuration over a 2, 2, 1.Thanks!    - 
[Jason Brownlee](https://machinelearningmastery.com) December 12, 2017 at 4:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-422877 "Direct link to this comment")The final model has the shape \[7, 5, 3\]. Perhaps check the tutorial again? It has 3 outputs, one for each of the 3 classes in the dataset.Configuration was chosen via trial and error. There is no analytical way to choose a configuration for a neural network.Finally, you can learn more about array indexing in Python here:\
 [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)93. 
Mohamed Elshazly December 17, 2017 at 9:34 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-423629 "Direct link to this comment")Hi Jason In the tain\_network function the line “expected\[row\[-1\]\] \= 1” what i understand is that you take the Y value of every row (which is either 0 or 1 ) and use it as an index in the expected array and you change the value at that index to 1 ,First i don’t know if i understand that correctly in the first place or not but if so, Wouldn’t the modification to the expected array be locked down to just only the first and second index because “expected\[row\[-1\]\] \= 1” would only be expected\[0\] or expected\[1\] ? and how would that help in our algorithm . looking forward to your response and thanks for the Great Tutorial    - 
[Jason Brownlee](https://machinelearningmastery.com) December 18, 2017 at 5:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-423671 "Direct link to this comment")We are one hot encoding the variable.Learn more about it here:\
 [https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)94. 
MohamedElshazly December 18, 2017 at 4:55 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-423734 "Direct link to this comment")HI again Jason  If I’m implementing this algorithm in python 3 what should i change in expected\[row\[-1\]\]\=1 in order for it to work because I’m having this error : list assignment index out of range\
 thanks in advance    - 
[Jason Brownlee](https://machinelearningmastery.com) December 19, 2017 at 5:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-423779 "Direct link to this comment")I don’t know off the cuff, I will look into porting the example to Py3 in the new year.95. 
[Tushar](http://www.tusharahuja.in) December 19, 2017 at 5:29 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-423941 "Direct link to this comment")You are just awesome Jason. You are adding more value to people’s ML skills than most average graduate schools do in the US.Thanks a ton!    - 
[Jason Brownlee](https://machinelearningmastery.com) December 20, 2017 at 5:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-424039 "Direct link to this comment")Thanks Tushar.96. 
[mark](http://n.a) January 6, 2018 at 3:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-425739 "Direct link to this comment")Wow, thanks for your codes. I have a question, what if I want to add regularisation term like L2 during back propagation, what should i do?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 6, 2018 at 5:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-425753 "Direct link to this comment")I would recommend moving to a platform like Keras:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)        - 
mark January 7, 2018 at 5:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-425829 "Direct link to this comment")Thanks for replying. I know the keras and have been using keras for a while. But in the problem I am focusing on, I need to make changes on the back propagation. That’s why I didn’t use keras.\
 So let’s go back to my original question, is the error term the cost function? Thanks.            - 
[Jason Brownlee](https://machinelearningmastery.com) January 8, 2018 at 5:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-425849 "Direct link to this comment")Sorry, I cannot work-through adding regularization to this tutorial for you.97. 
Mojo January 20, 2018 at 7:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-427014 "Direct link to this comment")Hello Jeson,\
 Thanks for the informative tutorial. I have a question.\
 if i want to change the error equation and as well as the equation between input with hidden and hidden with output layer. How can i change it?\
 Hope you will reply in a short time.Regards,\
 Mojo    - 
[Jason Brownlee](https://machinelearningmastery.com) January 21, 2018 at 9:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-427048 "Direct link to this comment")I would recommend using a library instead, it will be much easier for you.Here’s how to get started with Keras:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)98. 
Aliya Anil February 16, 2018 at 9:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-429705 "Direct link to this comment")Hi Jason,It was indeed a very informative tutorial. Could you please explain the need for seed(1) in the code?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 17, 2018 at 8:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-429751 "Direct link to this comment")I am trying to tie down the random number generator so that you get the same results as me.Learn more here:\
 [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)99. 
Raj February 19, 2018 at 7:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-429886 "Direct link to this comment")Hey there,\
 Been following your tutorial and I’m having problems with using my dataset with it. The outputs of the hidden neurons appear to only be exactly 1 constantly. I’m not sure what’s wrong exactly or how to fix it but its resulting in the network not learning at all. Please let me know if you can help.\
 Thanks,\
 Raj    - 
[Jason Brownlee](https://machinelearningmastery.com) February 19, 2018 at 9:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-429909 "Direct link to this comment")Perhaps try to get the code and data in the tutorial working first and use that as a starting point for your own problem.Generally, I would recommend using a library like Keras for your own projects and only code methods from scratch as a learning exercise.100. 
Aliya Anil February 20, 2018 at 4:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430009 "Direct link to this comment")Hi,I tried the first code in the tutorial with 4-parameter dataset, but it is not predicting like the 2-parameter set. Could you explain the reason?Thanks,\
 Aliya    - 
[Jason Brownlee](https://machinelearningmastery.com) February 21, 2018 at 6:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430097 "Direct link to this comment")If you are looking to develop a neural network for your own data, I would recommend the Keras library:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)101. 
Nik March 2, 2018 at 1:54 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430881 "Direct link to this comment")Dear Jason,Can I use the codes for handwritten digits recognition? If yes, are there any special recommendations what to change in the codes or I can use them with no changes?Thanks,\
 Nik    - 
[Jason Brownlee](https://machinelearningmastery.com) March 2, 2018 at 3:25 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430894 "Direct link to this comment")I would recommend this tutorial:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)        - 
Nik March 2, 2018 at 7:51 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430908 "Direct link to this comment")Yes, I have seen that tutorial. But is there any way to use the codes from this tutorial? I just would like to understand why they work so well for seeds and do not work for handwritten digits…            - 
[Jason Brownlee](https://machinelearningmastery.com) March 3, 2018 at 8:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-430959 "Direct link to this comment")Yes, you can. You can develop a model with all pixels as inputs.I cannot write the modification for you, sorry, I just don’t have the capacity.102. 
Filoingko March 4, 2018 at 2:58 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431046 "Direct link to this comment")Hi, How can I use this trained network to predict another data set. Thank you.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 4, 2018 at 6:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431076 "Direct link to this comment")The code in this tutorial is to teach you about backprop, not for use on real problems. If you are working through a problem, I’d recommend using Keras.103. 
[Jean-Michel Richer](http://www.info.univ-angers.fr/~richer/) March 5, 2018 at 9:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431224 "Direct link to this comment")Dear Jason,\
 I have tried to use your code on a simple XOR example but get a result of \[0, 0, 1, 1\] instead of \[0,1,1,0\]\
 Scores: \[0.0\]\
 Mean Accuracy: 0.000%The input xor.csv file is\
 0,0,0\
 0,1,1\
 1,0,1\
 1,1,0For this I have modified the evaluate\_algorithm function to:\
 def evaluate\_algorithm\_no\_fold(dataset, algorithm, \*args):\
 scores \= list()\
 predicted \= algorithm(dataset, dataset, \*args)\
 print(predicted)\
 accuracy \= accuracy\_metric(dataset, predicted)\
 scores.append(accuracy)\
 return scoresand call the function like this:\
 scores \= evaluate\_algorithm\_no\_fold(dataset, back\_propagation, 0.1, 500, 4)Would you have some explanation because I can not figure out why it is not working ?\
 Best regards,\
 JM    - 
[Jason Brownlee](https://machinelearningmastery.com) March 6, 2018 at 6:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431276 "Direct link to this comment")Perhaps the model requires tuning to your new dataset.104. 
Tanveer March 5, 2018 at 9:29 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431225 "Direct link to this comment")Thank You So Much Jason !! Wonderful Tutorial. THANKS Much !!    - 
[Jason Brownlee](https://machinelearningmastery.com) March 6, 2018 at 6:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431277 "Direct link to this comment")You’re welcome.105. 
Mojo March 9, 2018 at 10:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431697 "Direct link to this comment")If i want to calculate the training accuracy and F-measure and want to change the activation function, how i can do it?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 10, 2018 at 6:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431742 "Direct link to this comment")Perhaps you would be better off using scikit-learn and Keras instead.106. 
Fahad March 12, 2018 at 8:03 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431941 "Direct link to this comment")Is there something wrong with this code in case of using MINIST data? I tried to change the structured of the data to be compatible with the code, but it gave me a huge error and the error did not decrees during all training steps    - 
[Jason Brownlee](https://machinelearningmastery.com) March 13, 2018 at 6:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-431970 "Direct link to this comment")The code was not developed for MNIST. Here is an example of working with MNIST:\
 [https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/](https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)107. 
Fahad March 13, 2018 at 4:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432017 "Direct link to this comment")Thanks Jason for your response. I want to apply the code without keras. I tried to change the structure of the data to be each row as a vector of 784 pixel followed by a class label, but as I said it gave a huge error and does not decrees at all. I am trying to develop some algorithm for enhancing of learning, hence, I need to deal with the procedure as step by step. So keras or any other library does not help.Thanks again Jason    - 
[Jason Brownlee](https://machinelearningmastery.com) March 14, 2018 at 6:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432092 "Direct link to this comment")Perhaps update the code to use numpy, it will be much faster.108. 
kelvin March 15, 2018 at 2:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432155 "Direct link to this comment")Hi Mr Brownlee,Can you teach me how to plot the errors per epochs (validation error) and accuracy for both training and validation in your scratch network?    - 
kelvin March 15, 2018 at 2:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432157 "Direct link to this comment")I only can find the training error but not validation error in the code. For the accuracy, I plot a graph have a straight line only.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 15, 2018 at 6:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432177 "Direct link to this comment")Yes, you can do it easily in Keras here:\
 [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)        - 
kelvin March 15, 2018 at 12:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432204 "Direct link to this comment")Is there any possible way to do it on your scratch network? for example which part of the code save the training error, validation error, training accuracy and validation accuracy? So I can plot the graph myself since your scratch model does not have “model” for me to save the history.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 15, 2018 at 2:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432223 "Direct link to this comment")Yes, perhaps change it from CV to a single/train test, then evaluate the model skill on each dataset at the end of each epoch. Save the results in a list and return the lists.        - 
Zahra May 6, 2019 at 9:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484198 "Direct link to this comment")Hello, I’m so confuse..\
 I try to run this code in command prompt. But, I use my dataset (not Wheat Seeds dataset).And why this happened? What’s wrong? What should I do? What should I change?\
 Please, help me! ????????????????Traceback (most recent call last):\
 File “journal.py”, line 197, in\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 File “journal.py”, line 81, in evaluate\_algorithm\
 predicted \= algorithm(train\_set, test\_set, \*args)\
 File “journal.py”, line 173, in back\_propagation\
 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 File “journal.py”, line 150, in train\_network\
 expected\[row\[-1\]\] \= 1\
 IndexError: list assignment index out of range            - 
[Jason Brownlee](https://machinelearningmastery.com) May 6, 2019 at 2:33 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484218 "Direct link to this comment")Sorry, I cannot debug your dataset.Perhaps start with Keras for deep learning instead:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)109. 
Jack March 15, 2018 at 12:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432202 "Direct link to this comment")Can I use this model for regression problem? For example us this model for boston house-prices dataset?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 15, 2018 at 2:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432222 "Direct link to this comment")Sure, some changes would be required, such as the activation in the output layer would need to be linear.110. 
Nabil March 15, 2018 at 3:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432230 "Direct link to this comment")Are you using MSE?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 16, 2018 at 6:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432288 "Direct link to this comment")As mentioned in the post, we are reporting accuracy for the classification problem.111. 
Olu March 19, 2018 at 11:23 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432619 "Direct link to this comment")In the train section,12345678def train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs ) :for  epoch in  range ( n\_epoch ) :for  row in  train :outputs  \=  forward\_propagate ( network ,  row )expected  \=  \[ 0  for  i  in  range ( n\_outputs ) \]expected \[ row \[ - 1 \] \]  \=  1backward\_propagate\_error ( network ,  expected )update\_weights ( network ,  row ,  l\_rate )Can you please explain how this expected\[row\[-1\]\] \= 1 knows where to insert the 1 in the arrays of zero created.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 20, 2018 at 6:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432666 "Direct link to this comment")Good question.expected is all zeros. row\[-1\] is the index of the class value. therefore we set the index of the class value in expected to 1.Perhaps it is worth reading up on array indexing:\
 [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)        - 
kmillen November 16, 2018 at 11:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454958 "Direct link to this comment")Jason,\
 This is an amazing piece of code that has been very beneficial.Doesn’t that mean that only expected\[0\] and expected\[1\] will ever be set to 1 for this test data?Thank you,            - 
[Jason Brownlee](https://machinelearningmastery.com) November 16, 2018 at 1:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454973 "Direct link to this comment")Sorry, I don’t understand your question?                - 
kmillen November 17, 2018 at 1:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455042 "Direct link to this comment")If I understand Python (which I may not), row\[-1\] represents the last item in the row. Since the last value in each of the 10 rows is only either 0 or 1, expected\[row\[-1\]\] \= 1 will only ever set expected\[0\] or expected\[1\] to the value of 1. Or, what am I missing?                - 
[Jason Brownlee](https://machinelearningmastery.com) November 17, 2018 at 5:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455079 "Direct link to this comment")Are you referring to this: expected\[row\[-1\]\] \= 1If so:“expected” is all zeros, e.g. \[0, 0\]\
 “row” is an example, e.g. \[…\] where the value at -1 is either 0 or 1Therefore row\[-1\] is an index of either 0 or 1 and we are marking the value in expected at that index as 1.We have created a one hot vector.            - 
kmillen November 17, 2018 at 2:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455048 "Direct link to this comment")Disregard my previous question; I found the answer in a previous reply. Thank you again for this example.112. 
kelvin March 21, 2018 at 2:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432741 "Direct link to this comment")Hi, I would like to use softmax as the activation function for output layer. However, I do not know how to write the code for the derivative of softmax. Can you show me the code how to change the sigmoid function from your code to softmax?    - 
kelvin March 21, 2018 at 2:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432742 "Direct link to this comment")I do try few ways to change the sigmoid to softmax, however, all of them are not working. Can you show me how to create a softmax layer?for transfer():\
 first case:\
 def transfer(input\_value):\
 exp\_scores \= np.exp(input\_value)\
 return exp\_scores / np.sum(exp\_scores, axis\=1, keepdims\=True)second case:\
 def transfer(input\_value):\
 input\_value -\= np.max(input\_value)\
 return np.exp(input\_value) / np.sum(np.exp(input\_value))third case:\
 def transfer(input\_value):\
 input\_value -\= np.max(input\_value)\
 result \= (np.exp(input\_value).T / np.sum(np.exp(input\_value))).T\
 return resultfor transfer\_derivative():\
 first case:\
 def transfer\_derivative(output):\
 s \= output.reshape(-1, 1)\
 return np.diagflat(s) – np.dot(s, s.T)second case:\
 def transfer\_derivative(output):\
 jacobian\_m \= np.diag(output)\
 for i in range(len(jacobian\_m)):\
 for j in range(len(jacobian\_m)):\
 if i \=\= j:\
 jacobian\_m\[i\]\[j\] \= output\[i\] \* (1 – output\[i\])\
 else:\
 jacobian\_m\[i\]\[j\] \= -output\[i\] \* output\[j\]\
 return jacobian\_m        - 
[Jason Brownlee](https://machinelearningmastery.com) March 21, 2018 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432771 "Direct link to this comment")Here you go:\
 [https://en.wikipedia.org/wiki/Softmax\_function#Artificial\_neural\_networks](https://en.wikipedia.org/wiki/Softmax_function#Artificial_neural_networks)    - 
[Jason Brownlee](https://machinelearningmastery.com) March 21, 2018 at 6:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-432770 "Direct link to this comment")Perhaps use Keras instead?113. 
Suede March 29, 2018 at 6:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-433545 "Direct link to this comment")hey Jason, this is very helpful. I have run the code but i keep on getting this error, can you please help me out? the error is:NameError Traceback (most recent call last)\
 in ()\
 186 str\_column\_to\_float(dataset, i)\
 187 # convert class column to integers\
 –> 188 str\_columnto\_int(dataset, len(dataset\[0\])-1)\
 189 # normalize input variables\
 190 minmax \= dataset\_minmax(dataset)NameError: name ‘str\_columnto\_int’ is not defined    - 
[Jason Brownlee](https://machinelearningmastery.com) March 29, 2018 at 6:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-433550 "Direct link to this comment")The code was written for Python 2.7, confirm that you are using this version of Python?114. 
[Fahri Güreşçi](http://fahriguresci.com) April 15, 2018 at 7:03 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435047 "Direct link to this comment")The csv file is not working. Edited csv file > bit.ly/2GYX2dF\
 you can use python 2 or 3\
 results:\
 python2 > Mean Accuracy: 95.238%\
 python3 > Mean Accuracy: 93.333%Why different?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 16, 2018 at 6:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435097 "Direct link to this comment")Wh is the CSV file not working? The link appears to work fine.Good question re the difference, no idea. Perhaps small differences in the API? The code was written for Py2, so it may require changes for py3.Also, see this post on the stochastic nature of ml algorithms:\
 [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)115. 
Fahad April 18, 2018 at 8:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435339 "Direct link to this comment")I have altered the code to work with MNIST (digit numbers) , the problem I have faced that forward\_propagate function returns \[1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 \] for each instance !Any help    - 
[Jason Brownlee](https://machinelearningmastery.com) April 19, 2018 at 6:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435390 "Direct link to this comment")Well done.The model will require tuning for the problem.116. 
Fahad April 19, 2018 at 7:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435408 "Direct link to this comment")Could you explain more in some details please.    - 
[Jason Brownlee](https://machinelearningmastery.com) April 19, 2018 at 2:45 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435428 "Direct link to this comment")What details exactly?117. 
Fahad April 19, 2018 at 8:17 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435446 "Direct link to this comment")As I mentioned that forward\_propagation function returns \[1,1,1,1,1,1,1,1,1,1\], what is the possible alter to come over this problem    - 
[Jason Brownlee](https://machinelearningmastery.com) April 20, 2018 at 5:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435477 "Direct link to this comment")Perhaps tune the model to your specific problem. I have some suggestions here:\
 [https://machinelearningmastery.com/improve-deep-learning-performance/](https://machinelearningmastery.com/improve-deep-learning-performance/)118. 
Fahad April 23, 2018 at 5:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435665 "Direct link to this comment")I altered the code to work with XOR problem and it was working perfectly. Then, I altered the code to work with digit numbers MNIST, but as I told you there was a problem with the the forward\_propagation function that it returned all outputs to be \[1,1,1,…\] instead of a probabilities for each output.\
 I think it is not an optimization problem, there is something wrong with the forward\_propagate function.Here is the code after alteration \[it is working but with a fixed error during all training epochsfrom random import seed\
 from random import randrange\
 from random import random\
 from csv import reader\
 from math import exp\
 global network\
 global gl\_errors# Load a CSV file\
 def load\_csv(filename):\
 dataset \= list()\
 with open(filename, ‘r’) as file:\
 csv\_reader \= reader(file)\
 for row in csv\_reader:\
 if not row:\
 continue\
 dataset.append(row)\
 return dataset# Convert string column to float\
 def str\_column\_to\_float(dataset, column):\
 for row in dataset:\
 row\[column\] \= float(row\[column\].strip())def str\_column\_to\_intX(dataset, column):\
 for row in dataset:\
 row\[column\] \= int(row\[column\].strip())# Convert string column to integer\
 def str\_column\_to\_int(dataset, column):\
 class\_values \= \[row\[column\] for row in dataset\]\
 unique \= set(class\_values)\
 lookup \= dict()\
 for i, value in enumerate(unique):\
 lookup\[value\] \= i\
 for row in dataset:\
 row\[column\] \= lookup\[row\[column\]\]\
 return lookup# Find the min and max values for each column\
 def dataset\_minmax(dataset):\
 minmax \= list()\
 stats \= \[\[min(column), max(column)\] for column in zip(\*dataset)\]\
 return stats# Rescale dataset columns to the range 0-1\
 def normalize\_dataset(dataset):\
 for row in dataset:\
 for i in range(1,len(row)):\
 # row\[i\] \= (row\[i\] – minmax\[i\]\[0\]) / (minmax\[i\]\[1\] – minmax\[i\]\[0\])\
 if row\[i\]>10:\
 row\[i\]\=1\
 else:\
 row\[i\]\=0# Split a dataset into k folds\
 def cross\_validation\_split(dataset, n\_folds):\
 dataset\_split \= list()\
 dataset\_copy \= list(dataset)\
 fold\_size \= int(len(dataset) / n\_folds)\
 for i in range(n\_folds):\
 fold \= list()\
 while len(fold) epoch\=%d, lrate\=%.3f, error\=%.3f’ % (epoch, l\_rate, sum\_error))# Calculate neuron activation for an input\
 def activate(weights, inputs): activation \= weights\[-1\]\
 for i in range(len(weights)-1):\
 activation \+\= weights\[i\] \* inputs\[i\]\
 return activation# Transfer neuron activation\
 def transfer(activation):\
 return 1.0 / (1.0 \+ exp(-activation))# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\[1:\]\
 i\=0 for layer in network:\
 new\_inputs \= \[\]\
 i\+\=1\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs return inputs# Calculate the derivative of an neuron output\
 def transfer\_derivative(output):\
 return output \* (1.0 – output)# Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 # err \=0\
 for i in reversed(range(len(network))):\
 layer \= network\[i\]\
 errors \= list()\
 if i !\= len(network)-1:\
 for j in range(len(layer)):\
 error \= 0.0\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 errors.append(error)\
 else:\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 errors.append(expected\[j\] – neuron\[‘output’\]) for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])# Update network weights with error\
 def update\_weights(network, row, l\_rate): for i in range(len(network)):\
 inputs \= row\[1:\]\
 if i !\= 0:\
 inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]\
 for neuron in network\[i\]:\
 for j in range(len(inputs)):\
 neuron\[‘weights’\]\[j\] \+\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] \+\= l\_rate \* neuron\[‘delta’\] hidden\_layer2 \= \[{‘weights’:\[random() for i in range(n\_hidden \+ 1)\]} for i in range(100)\]\
 network.append(hidden\_layer2) hidden\_layer3 \= \[{‘weights’:\[random() for i in range(100 \+ 1)\]} for i in range(50)\]\
 network.append(hidden\_layer3) output\_layer \= \[{‘weights’:\[random() for i in range(50 \+ 1)\]} for i in range(n\_outputs)\]\
 network.append(output\_layer)\
 return network# Make a prediction with a network\
 def predict(network, row):\
 outputs \= forward\_propagate(network, row)\
 return outputs.index(max(outputs)) # Test Backprop on Seeds dataset\
 seed(1)\
 # load and prepare data\
 filename \= ‘dataset/train2.csv’\
 dataset \= load\_csv(filename)for i in range(1,len(dataset\[0\])):\
 str\_column\_to\_float(dataset, i)# convert class column to integers\
 str\_column\_to\_int(dataset, 0)normalize\_dataset(dataset)# evaluate algorithm\
 n\_folds \= 5\
 l\_rate \= 0.5\
 n\_epoch \= 100\
 n\_hidden \= 500scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 print(‘Scores: %s’ % scores)\
 print(‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores))))    - 
[Jason Brownlee](https://machinelearningmastery.com) April 23, 2018 at 6:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435681 "Direct link to this comment")Sorry, I don’t have the capacity to debug the modified code for you.119. 
Rahmad ars April 26, 2018 at 3:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435926 "Direct link to this comment")Sir, can you help me?\
 this is my question..[https://stackoverflow.com/questions/50027886/need-help-for-check-my-backprop-ann-using-python](https://stackoverflow.com/questions/50027886/need-help-for-check-my-backprop-ann-using-python)    - 
[Jason Brownlee](https://machinelearningmastery.com) April 26, 2018 at 6:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435970 "Direct link to this comment")Perhaps you can summarize your question in a sentence or two?120. 
Rahmad ars April 26, 2018 at 7:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435975 "Direct link to this comment")your original code sir, only have 1 hidden layer with 2 neurons. then, I modify it, so the ANN have 3 hidden layers, each consist of (128, 64, 32). and i have my own dataset, so i change it (the dataset and input neurons). when i run this code, everything looks fine but the error value is not changing…here’s the screen: [https://i.stack.imgur.com/NQbNd.png](https://i.stack.imgur.com/NQbNd.png)modified code: [https://stackoverflow.com/questions/50027886/need-help-for-check-my-backprop-ann-using-python](https://stackoverflow.com/questions/50027886/need-help-for-check-my-backprop-ann-using-python)thanks sir    - 
[Jason Brownlee](https://machinelearningmastery.com) April 26, 2018 at 2:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-435997 "Direct link to this comment")If you are struggling with the code, I would recommend not coding the algorithm from scratch.Instead, I would recommend using a library like Keras. Here is a worked example:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)121. 
Fahad April 28, 2018 at 9:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436195 "Direct link to this comment")I have the same problem of RahmadThe same problem occurs when you change the original code from 5 neurons in the hidden layer to 31 neuron ( the error value does not change). I know 31 hidden neuron is not a right number of neurons for seed data set. But I would like to know what is the wrong when you increase the number of neurons.Logically, it should be fine and the error value decreases. But when you change the number of neurons to 30 it is still working , when change it to 31 neurons it does not decrease !I think if this problem is fixed, then the problem of Rahmad will be fixed too.    - 
[Jason Brownlee](https://machinelearningmastery.com) April 29, 2018 at 6:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436225 "Direct link to this comment")Perhaps the model requires tuning to your specific problem (e.g. layers, nodes, activation function, etc.)It might be better to use a library like Keras for your project:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)122. 
Rocha May 2, 2018 at 9:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436476 "Direct link to this comment")Hi dude, I’m stuck in this error, could you help me?# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\
 for layer in network:\
 new\_inputs \= \[\]\
 for neuron in layer:\
 —>>> activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 return inputsThat line is giving me this: TypeError: list indices must be integers or slices, not strShould be the python version? I’m using python 3…    - 
[Jason Brownlee](https://machinelearningmastery.com) May 3, 2018 at 6:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436513 "Direct link to this comment")This code was written for Python 2.7 sorry.123. 
Kamrun Nahar Nisha May 8, 2018 at 4:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436927 "Direct link to this comment")hello.please help me.\
 I want to use breast cancer dataset instead of seed dataset. seed(1)\
 # load and prepare data\
 filename \= ‘seeds\_dataset.csv’\
 dataset \= load\_csv(filename)\
 for i in range(len(dataset\[0\])-1):\
 str\_column\_to\_float(dataset, i)\
 # convert class column to integers\
 str\_column\_to\_int(dataset, len(dataset\[0\])-1)\
 # normalize input variables\
 minmax \= dataset\_minmax(dataset)\
 normalize\_dataset(dataset, minmax)\
 # evaluate algorithm\
 n\_folds \= 5\
 l\_rate \= 0.3\
 n\_epoch \= 500\
 n\_hidden \= 5\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 print(‘Scores: %s’ % scores)\
 print(‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores))))In this part of your code I also want to print the error and it will be likeepoch\=0, lrate\=0.500, error\=6.350\
 >epoch\=1, lrate\=0.500, error\=5.531\
 >epoch\=2, lrate\=0.500, error\=5.221\
 >epoch\=3, lrate\=0.500, error\=4.951\
 >epoch\=4, lrate\=0.500, error\=4.519\
 >epoch\=5, lrate\=0.500, error\=4.173\
 >epoch\=6, lrate\=0.500, error\=3.835\
 >epoch\=7, lrate\=0.500, error\=3.506\
 >epoch\=8, lrate\=0.500, error\=3.192\
 >epoch\=9, lrate\=0.500, error\=2.898\
 >epoch\=10, lrate\=0.500, error\=2.626\
 >epoch\=11, lrate\=0.500, error\=2.377\
 >epoch\=12, lrate\=0.500, error\=2.153\
 >epoch\=13, lrate\=0.500, error\=1.953\
 >epoch\=14, lrate\=0.500, error\=1.774\
 >epoch\=15, lrate\=0.500, error\=1.614\
 >epoch\=16, lrate\=0.500, error\=1.472\
 >epoch\=17, lrate\=0.500, error\=1.346\
 >epoch\=18, lrate\=0.500, error\=1.233\
 >epoch\=19, lrate\=0.500, error\=1.132\
 \[{‘output’: 0.029980305604426185, ‘weights’: \[-1.4688375095432327, 1.850887325439514, 1.0858178629550297\], ‘delta’: -0.0059546604162323625}, {‘output’: 0.9456229000211323, ‘weights’: \[0.37711098142462157, -0.0625909894552989, 0.2765123702642716\], ‘delta’: 0.0026279652850863837}\]\
 \[{‘output’: 0.23648794202357587, ‘weights’: \[2.515394649397849, -0.3391927502445985, -0.9671565426390275\], ‘delta’: -0.04270059278364587}, {‘output’: 0.7790535202438367, ‘weights’: \[-2.5584149848484263, 1.0036422106209202, 0.42383086467582715\], ‘delta’: 0.03803132596437354}\]please tell me the code . Using breast cancer dataset not wheat seed dataset. I am not so good in coding that’s why I need your help immediately.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 9, 2018 at 6:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-436976 "Direct link to this comment")I’m eager to help, but I do not have the capacity to outline the changes or write the code for you.124. 
Akefar May 11, 2018 at 4:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437162 "Direct link to this comment")hi Jason,\
 I tried your code in my data set ,shape of my data is (576,16) .the problem is IndexError: list assignment index out of rangeis there any need to change your code for (576,16) data shape .\
 Thanks—————————————————————————\
 IndexError Traceback (most recent call last)\
 in ()\
 195 n\_epoch \= 500\
 196 n\_hidden \= 1\
 –> 197 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 198 print(‘Scores: %s’ % scores)\
 199 print(‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores)))) in evaluate\_algorithm(dataset, algorithm, n\_folds, \*args)\
 79 test\_set.append(row\_copy)\
 80 row\_copy\[-1\] \= None\
 —> 81 predicted \= algorithm(train\_set, test\_set, \*args)\
 82 actual \= \[row\[-1\] for row in fold\]\
 83 accuracy \= accuracy\_metric(actual, predicted) in back\_propagation(train, test, l\_rate, n\_epoch, n\_hidden)\
 171 n\_outputs \= len(set(\[row\[-1\] for row in train\]))\
 172 network \= initialize\_network(n\_inputs, n\_hidden, n\_outputs)\
 –> 173 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 174 predictions \= list()\
 175 for row in test: in train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 148 outputs \= forward\_propagate(network, row)\
 149 expected \= \[0 for i in range(n\_outputs)\]\
 –> 150 expected\[row\[-1\]\] \= 1\
 151 backward\_propagate\_error(network, expected)\
 152 update\_weights(network, row, l\_rate)IndexError: list assignment index out of range    - 
[Jason Brownlee](https://machinelearningmastery.com) May 11, 2018 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437186 "Direct link to this comment")You may need to change your data to match the model or the model to match the data.125. 
Pradeep May 14, 2018 at 3:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437427 "Direct link to this comment")Hi Jason, I tried your code on the same sample dataset, i am getting the following type error in the function activate. I am doing it in python3.6. hope to hear from you soonTraceback (most recent call last):\
 File “neural\_network.py”, line 94, in\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 File “neural\_network.py”, line 76, in train\_network\
 outputs \= forward\_propagate(network, row)\
 File “neural\_network.py”, line 31, in forward\_propagate\
 activation \= activate(neuron\[‘weights’\], inputs)\
 File “neural\_network.py”, line 18, in activate\
 activation \+\= weights\[i\] \* inputs\[i\]\
 TypeError: can’t multiply sequence by non-int of type ‘float’    - 
[Jason Brownlee](https://machinelearningmastery.com) May 14, 2018 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437458 "Direct link to this comment")The code requires Python 2.7.126. 
Deepak D May 17, 2018 at 6:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437787 "Direct link to this comment")Hi Jason Brownlee,I tried your code and experienced the some error applying the Backpropagation algorithm to the wheat seeds dataset. I am using python 2.7.Error type:File “C:\Python27\programs\backpropagation.py”, line 186,\
 in str\_column\_to\_float(dataset, i)File “C:\Python27\programs\backpropagation.py”, line 22,\
 in str\_column\_to\_float row\[column\] \= float(row\[column\].strip())\
 ValueError: could not convert string to float:    - 
[Jason Brownlee](https://machinelearningmastery.com) May 18, 2018 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437829 "Direct link to this comment")I am sorry to hear that, I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)127. 
Dhanya Hegde May 19, 2018 at 1:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437901 "Direct link to this comment")Hey Jason! Great work. Really helpful. I didn’t understand one part of your code. On what basis does predict function return the predicted value as 0 or 1, after taking the maximum of the two output neuron values?    - 
[Jason Brownlee](https://machinelearningmastery.com) May 19, 2018 at 7:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-437930 "Direct link to this comment")The summation of the activation is passed through a sigmoid transfer function.        - 
[Dhanya Hegde](http://.) May 21, 2018 at 3:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438072 "Direct link to this comment")I didn’t understand this part of the codeoutputs.index(max(outputs)Is one hot encoding used or binary classification?\
 If so, how is the actual mapping done?\
 And when is the iteration process stopped?            - 
[Jason Brownlee](https://machinelearningmastery.com) May 21, 2018 at 6:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438103 "Direct link to this comment")As stated in the text above the code, it returns an integer for the class with the largest probability.128. 
Ionut May 27, 2018 at 12:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438755 "Direct link to this comment")Hi,I’m a beginner in neural networks and I don’t understand the dataset from the section “4.2. Train Network”. Can anyone explain me what x1, x2 and y means?    - 
[Jason Brownlee](https://machinelearningmastery.com) May 27, 2018 at 6:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438786 "Direct link to this comment")Input 1, input 2 and output.Perhaps start here:\
 [https://machinelearningmastery.com/start-here/#algorithms](https://machinelearningmastery.com/start-here/#algorithms)129. 
Rishik Mani May 28, 2018 at 7:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438884 "Direct link to this comment")Hi Jason, thank you for the highly informative post. But could you please clarify me upon this petty little issue.In section 4.2 Train network, you considered n\_inputs \= len(dataset\[0\]) – 1. Why did you put a -1 here, while the number of the inputs should exactly be of the length of the dataset.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 28, 2018 at 2:32 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-438914 "Direct link to this comment")To exclude the output variable from the number of inputs.130. 
Samih Eisa June 2, 2018 at 9:13 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439536 "Direct link to this comment")Thank you, jason.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 3, 2018 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439591 "Direct link to this comment")You’re welcome.131. 
[Kie Woo Nam](https://github.com/gknam) June 5, 2018 at 3:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439801 "Direct link to this comment")12345678910# Update network weights with errordef update\_weights ( network ,  row ,  l\_rate ) :for  i  in  range ( len ( network ) ) :inputs  \=  row \[ : - 1 \]if  i  !\=  0 :inputs  \=  \[ neuron \[ 'output' \]  for  neuron in  network \[ i  -  1 \] \]for  neuron in  network \[ i \] :for  j  in  range ( len ( inputs ) ) :neuron \[ 'weights' \] \[ j \]  \+\=  l\_rate \*  neuron \[ 'delta' \]  \*  inputs \[ j \]neuron \[ 'weights' \] \[ - 1 \]  \+\=  l\_rate \*  neuron \[ 'delta' \]Hi,I guess I’m likely mistaken, so please but when i !\= 0, isn’t the last line updating the last weight for the second time?So, shouldn’t it be “inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]\[:-1\]” (add “\[:-1\]\]” at the end)?If I’m wrong, I’ll read the code again more carefully, so please let me know.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 5, 2018 at 6:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439837 "Direct link to this comment")No. There are more weights than inputs and the -1 index of the weights is the bias.        - 
[Kie Woo Nam](https://github.com/gknam) June 5, 2018 at 7:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439891 "Direct link to this comment")Ah, right. Now I see it from “output\_layer \= \[{‘weights’:\[random() for i in range(n\_hidden \+ 1)\]} for i in range(n\_outputs)\]”.Thank you for your quick reply.            - 
[Jason Brownlee](https://machinelearningmastery.com) June 6, 2018 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-439939 "Direct link to this comment")No problem.132. 
Thomas Specht July 10, 2018 at 4:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-443058 "Direct link to this comment")Hi Jason,Great tutorial to get into ML coding. I have one question:What library would you recommend for projects and why? I want to use NN for regression problems.    - 
[Jason Brownlee](https://machinelearningmastery.com) July 10, 2018 at 6:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-443091 "Direct link to this comment")I recommend Keras because it is computationally efficient, fast for development and fun:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)133. 
Hugo B. August 13, 2018 at 1:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-445938 "Direct link to this comment")Hi Jason! Thank you for this tutorial.\
 I try to implement the batch learning, but I have some questions about it…– Computing the errors:\
 Do I have to accumulate the errors (‘delta’) in backward\_propagate\_error() during one epoch and performing an average according to the number of back-propagations performed?– Updating the weights:\
 In train\_network(), I call update\_weights() for each epoch, but I don’t know which row(s) of train (dataset) I have to used. Currently I use only one row: train\[0\].134. 
elizabeth August 14, 2018 at 2:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-446016 "Direct link to this comment")Hello Sir\
 Thank you so much for this article.\
 Can you please tell me how i can solve this error: File “mlp8.py”, line 186, in\
 str\_column\_to\_float(dataset, i)\
 File “mlp8.py”, line 22, in str\_column\_to\_float\
 row\[column\] \= float(row\[column\].strip())\
 ValueError: could not convert string to float:thankyou    - 
[Jason Brownlee](https://machinelearningmastery.com) August 14, 2018 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-446048 "Direct link to this comment")I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)135. 
STEPHEN OLADEJI August 28, 2018 at 10:35 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-447148 "Direct link to this comment")Dear Prof,Thanks for this tutorial,\
 Sir, I’ve gone through a lot of your project online and they are very superb. God bless you, sir.\
 Sir. my question is as followings.\
 1. I noticed that one WEKA 3.6, Artificial Immune System was removed because this was in version 1.8, Is it because there is no research prospect in the algorithm?\
 2. I want to write python version for AIRS, CSCA, Genetic Algorithm can you help proofread it sir so see if what I write is correct    - 
[Jason Brownlee](https://machinelearningmastery.com) August 29, 2018 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-447185 "Direct link to this comment")I have an implementation here that you can use:\
 [http://wekaclassalgos.sourceforge.net/](http://wekaclassalgos.sourceforge.net/)136. 
Tamara September 11, 2018 at 5:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-448350 "Direct link to this comment")Thank you so much for this tutorial.\
 How to see the results of work a trained network in Python?    - 
[Jason Brownlee](https://machinelearningmastery.com) September 11, 2018 at 6:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-448368 "Direct link to this comment")What do you mean exactly?137. 
ritu September 23, 2018 at 9:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-449612 "Direct link to this comment")How should I modify to code to always run for one output neuron in the output layer?\
 eg. if the output class consist of only 2 output classes ‘1’ and ‘2’ , as per the above code 2 neurons will be created within the output layer, but what if I wanted a neural network to just have one neuron in the output layer.    - 
[Jason Brownlee](https://machinelearningmastery.com) September 24, 2018 at 6:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-449654 "Direct link to this comment")If you are having trouble with this tutorial, I would encourage you to use Keras to develop neural network models.You can get started here:\
 [https://machinelearningmastery.com/start-here/#python](https://machinelearningmastery.com/start-here/#python)138. 
Parva September 27, 2018 at 9:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-449919 "Direct link to this comment")Why is there just one output coming from layer 1 though it contains 2 neurons. Shouldn’t there be 2 outputs one from each neuron? \[{‘output’: 0.7105668883115941, ‘weights’: \[0.13436424411240122, 0.8474337369372327, 0.763774618976614\], ‘delta’: -0.0005348048046610517}\]\
 \[{‘output’: 0.6213859615555266, ‘weights’: \[0.2550690257394217, 0.49543508709194095\], ‘delta’: -0.14619064683582808}, {‘output’: 0.6573693455986976, ‘weights’: \[0.4494910647887381, 0.651592972722763\], ‘delta’: 0.0771723774346327}\]    - 
[Jason Brownlee](https://machinelearningmastery.com) September 27, 2018 at 2:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-449941 "Direct link to this comment")Which step exactly are you having trouble with?        - 
Brian September 28, 2018 at 1:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-449974 "Direct link to this comment")Thank you for this example. It has helped me get past the block I had with the mathematical based descriptions and differential calculus related to back propagation.\
 As I have been focusing on the back propagation portion of this example I have come up with an alternative version of the ‘backward\_propagate\_error’ function that I think is a much more succinct and logical way to write this function.Please find below1234567891011121314# Backpropagate error and store in neuronsdef backward\_propagate\_error ( network ,  expected ) :for  i  in  reversed ( range ( len ( network ) ) ) :layer  \=  network \[ i \]for  j  in  range ( len ( layer ) ) :fromNeuron  \=  layer \[ j \] error \= 0.0if  i  !\=  len ( network ) - 1 :  #This identifies all but the last (output) layerfor  toNeuron in  network \[ i  \+  1 \] :error  \+\=  ( toNeuron \[ 'weights' \] \[ j \]  \*  toNeuron \[ 'delta' \] ) else : #This is the last (output) layererror  \=  expected \[ j \]  -  fromNeuron \[ 'output' \]fromNeuron \[ 'error' \]  \=  errorfromNeuron \[ 'delta' \]  \=  error \*  transfer\_derivative ( fromNeuron \[ 'output' \] )            - 
[Jason Brownlee](https://machinelearningmastery.com) September 28, 2018 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-450000 "Direct link to this comment")Cool!Sorry, I don’t have the capacity to review your code.139. 
Brian October 4, 2018 at 2:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-450626 "Direct link to this comment")Should not the formula for the error be\
 sum\_error \+\= sum(\[0.5\*(expected\[i\]-outputs\[i\])\*\*2 for i in range(len(expected))\])\
 as apposed to\
 sum\_error \+\= sum(\[(expected\[i\]-outputs\[i\])\*\*2 for i in range(len(expected))\])\
 To correctly back propagate with the derivative?    - 
[Jason Brownlee](https://machinelearningmastery.com) October 4, 2018 at 6:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-450649 "Direct link to this comment")Why is that Brian?140. 
KS October 10, 2018 at 2:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451146 "Direct link to this comment")Can you please tell me how to inplant tanh and ReLu?\
 I tried to find examples on the internet, but I couldn’t find good examples.I understand that these 2 codes need to be changed:# Transfer neuron activation\
 def transfer(activation):\
 return 1.0 / (1.0 \+ exp(-activation))# Calculate the derivative of an neuron output\
 def transfer\_derivative(output):\
 return output \* (1.0 – output)1. What is the code when using tanh?\
 2. What is the code when using ReLu?    - 
[Jason Brownlee](https://machinelearningmastery.com) October 10, 2018 at 6:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451181 "Direct link to this comment")Thanks for the suggestion, sorry, I don’t have the capacity to make these changes for you.    - 
audrey April 14, 2020 at 5:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529639 "Direct link to this comment")did you find something about Relu?        - 
[Jason Brownlee](https://machinelearningmastery.com) April 14, 2020 at 6:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529676 "Direct link to this comment")See thus tutorial:\
 [https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)141. 
moe October 10, 2018 at 7:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451222 "Direct link to this comment")Thank you so much, 1 – 5 helped me to built my own generic network. Now i have a MLP network which i can easily adjust with a few parameter changes, wouldnt have been so easy with your example.    - 
[Jason Brownlee](https://machinelearningmastery.com) October 11, 2018 at 7:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451292 "Direct link to this comment")Well done!142. 
Anwar October 11, 2018 at 3:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451271 "Direct link to this comment")I am running the same code that u have provided on python 3.6 and getting these errors please help me:-—————————————————————————————————————————IndexError Traceback (most recent call last)\
 in ()\
 16 n\_epoch \= 500\
 17 n\_hidden \= 5\
 —> 18 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 19 print(‘Scores: %s’ % scores)\
 20 print(‘Mean Accuracy: %.3f%%’ % (sum(scores)/float(len(scores)))) in evaluate\_algorithm(dataset, algorithm, n\_folds, \*args)\
 12 test\_set.append(row\_copy)\
 13 row\_copy\[-1\] \= None\
 —> 14 predicted \= algorithm(train\_set, test\_set, \*args)\
 15 actual \= \[row\[-1\] for row in fold\]\
 16 accuracy \= accuracy\_metric(actual, predicted) in back\_propagation(train, test, l\_rate, n\_epoch, n\_hidden)\
 4 n\_outputs \= len(set(\[row\[-1\] for row in train\]))\
 5 network \= initialize\_network(n\_inputs, n\_hidden, n\_outputs)\
 —-> 6 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 7 predictions \= list()\
 8 for row in test: in train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 5 outputs \= forward\_propagate(network, row)\
 6 expected \= \[0 for i in range(n\_outputs)\]\
 —-> 7 expected\[row\[-1\]\] \= 1\
 8 backward\_propagate\_error(network, expected)\
 9 update\_weights(network, row, l\_rate)\
 IndexError: list assignment index out of range    - 
[Jason Brownlee](https://machinelearningmastery.com) October 11, 2018 at 8:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451307 "Direct link to this comment")I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)143. 
KS October 12, 2018 at 11:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451428 "Direct link to this comment")How to:\
 “Regression. Change the network so that there is only one neuron in the output layer and that a real value is predicted.”How to get 1 output layer?    - 
[Jason Brownlee](https://machinelearningmastery.com) October 13, 2018 at 6:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451512 "Direct link to this comment")Coding algorithms from scratch is not for beginners.I strongly encourage you to use Keras, for example:\
 [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)144. 
Jerry October 16, 2018 at 9:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451754 "Direct link to this comment")Ok, just following up. I still don’t get how the backprop is updating:1234567for  row in  train :outputs  \=  forward\_propagate ( network ,  row ,  alpha )expected  \=  \[ 0  for  i  in  range ( n\_outputs ) \]expected \[ row \[ - 1 \] \]  \=  1sum\_error  \+\=  sum ( \[ ( expected \[ i \] - outputs \[ i \] ) \* \* 2  for  i  in  range ( len ( expected ) ) \] )backward\_propagate\_error ( network ,  expected ,  alpha )update\_weights ( network ,  row ,  l\_rate )1. No return statement or ‘global network’ for backward\_propagate\_error or update\_network to actually incorporate the new weights. My question, are you sure this uses back-propagation? How are the weights saved and updated for each epoch?    - 
[Jason Brownlee](https://machinelearningmastery.com) October 16, 2018 at 2:34 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-451767 "Direct link to this comment")The weights are passed in by reference and modified in place.Perhaps using Keras would be a better fit Jerry:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)        - 
Jerry November 10, 2018 at 11:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454335 "Direct link to this comment")You learn something new everyday! I was always under the impression that values were immutable when passing to a function in Python. [https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference](https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference)Thank you Jason! I’m really amazed on how active you are on this site. Just so you know, you and your work are referenced often in my MSDS program. One area that I think would also be beneficial is some work on hidden node activations and their interpretations. The majority of our work is not so much on the output/accuracy of the NN, but more of visualizing weights, activations, and determining the features that are causing nodes to activate.            - 
[Jason Brownlee](https://machinelearningmastery.com) November 11, 2018 at 6:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454373 "Direct link to this comment")Thanks, happy to help.Why do you want to view/understand the dynamics of nodes in hidden layers?    - 
ben October 31, 2018 at 9:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-453276 "Direct link to this comment")i think this is not correct:\
 sum\_error \+\= sum(\[(expected\[i\]-outputs\[i\])\*\*2 for i in range(len(expected))\])shouldn´t this be:\
 sum\_error \+\= sum(\[abs(0.5\*(expected\[i\]-outputs\[i\])\*\*2) for i in range(len(expected))\])see [https://goo.gl/iqHJf6](https://goo.gl/iqHJf6) page 233 (5.11).        - 
[Jason Brownlee](https://machinelearningmastery.com) November 1, 2018 at 6:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-453308 "Direct link to this comment")This implementation is based on the book “Neural Smithing”:\
 [https://www.amazon.com/Neural-Smithing-Supervised-Feedforward-Artificial/dp/0262527014/ref\=as\_li\_ss\_tl?ie\=UTF8&linkCode\=sl1&tag\=inspiredalgor-20&linkId\=e3db0b57249093a94ebb073983bc8b4d&language\=en\_US](https://www.amazon.com/Neural-Smithing-Supervised-Feedforward-Artificial/dp/0262527014/ref=as_li_ss_tl?ie=UTF8&linkCode=sl1&tag=inspiredalgor-20&linkId=e3db0b57249093a94ebb073983bc8b4d&language=en_US)145. 
Ayhan October 20, 2018 at 1:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-452149 "Direct link to this comment")Hey Jason, first thnx for the efforts. Maybe one thing:\
 wouldnt it be better (especially for beginners to that topic) to at least use\
 something like numpy which would at least let all the matrix calculation look\
 a bit more compact (and therefore possible to concentrate on the real topic .. which is\
 backprop i would have guessed)?Ok the title sais ‘implement from scratch’ but I would say at some point getting\
 the point towards backprop is maybe more important than being able to say that\
 it was implemented from scratch (using nothing but plain python). Greets    - 
[Jason Brownlee](https://machinelearningmastery.com) October 20, 2018 at 5:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-452164 "Direct link to this comment")Thanks for the feedback.This example is not intended for writing operational code, for that I would recommend Keras:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)This tutorial is to teach how to develop a net (training and evaluation) without any dependencies other than the standard lib.146. 
Vadim November 3, 2018 at 11:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-453576 "Direct link to this comment")Just Brilliant stuff    - 
[Jason Brownlee](https://machinelearningmastery.com) November 4, 2018 at 6:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-453604 "Direct link to this comment")Thanks.147. 
Himanshu November 14, 2018 at 10:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454771 "Direct link to this comment")Hi Jason\
 how i can use forward and backward propagation in real time data. I mean data in which we have multiple field. Fields contains Numerical values , floating values , String Values.\
 For an example if i want to use this technique for Titanic data how can i use it.How can i decide how many hidden layers should be there.\
 How can i decide for learning rate.\
 How can i decide for what should be the seed value.\
 How can i decide for weights for such huge data.    - 
[Jason Brownlee](https://machinelearningmastery.com) November 15, 2018 at 5:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454801 "Direct link to this comment")You must use careful experimentation to get answers to each of these questions.148. 
Himanshu November 15, 2018 at 6:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454867 "Direct link to this comment")Hi Jason\
 I have few more questions on this.First: why you assigned “activation \= weights\[-1\]” , why not other weights or any random value.\
 Second: why you are looping only for two values\
 for i in range(len(weights) – 1) though we have three values.\
 Third: Why you have considered only two outputs here though we have four\
 output for layer one \=.9643898158763548\
 output for layer two \=.9185258960543243\
 output for layer three \= .8094918973879515\
 output for layer two \= .7734292563511262why you considered only last two values why not first two or any other combination.Four: here i guess some problem by mistake you have update wrong value\
 expected\[row\[-1\]\] \= 1\
 after this line you have updated expected as \[1,0\] from \[0,0\]\
 and why we in this i have other question why we are updating this value.    - 
[Jason Brownlee](https://machinelearningmastery.com) November 16, 2018 at 6:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-454925 "Direct link to this comment")Too many questions for one comment (I’m a simple human), one at a time and can you please reference specific examples/lines of code otherwise I can’t help.149. 
Karim November 18, 2018 at 8:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455251 "Direct link to this comment")Hi Jason — How can i add more hidden layers? Thankx    - 
[Jason Brownlee](https://machinelearningmastery.com) November 19, 2018 at 6:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455426 "Direct link to this comment")I recommend using Keras to develop your network:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)150. 
Himanshu November 22, 2018 at 5:24 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455842 "Direct link to this comment")Hi Jason,How to apply mathematical implementation of gradient descent and logistic regression,classification in real time data.\
 For example if i want use this in survivors of Titanic data how to start with.    - 
[Jason Brownlee](https://machinelearningmastery.com) November 23, 2018 at 7:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-455884 "Direct link to this comment")Here’s an example:\
 [https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)151. 
John Sald December 6, 2018 at 2:05 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-457560 "Direct link to this comment")Hello, could you show me an example of using one of the extensions you mentioned, which can give us a gain in performance?Such as using matrix operations (in the weights) and vectors (inputs, intermediate signals and outputs)    - 
[Jason Brownlee](https://machinelearningmastery.com) December 7, 2018 at 5:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-457691 "Direct link to this comment")Thanks for the suggestion.If you’re looking to go deeper into neural nets, I recommend using a library like Keras. You can start here:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)152. 
Pipo December 17, 2018 at 7:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-458835 "Direct link to this comment")How could i change the loss to mse in this code? I can’t wrap my head around it. Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) December 17, 2018 at 2:13 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-458865 "Direct link to this comment")Calculate error between actual and predicted using a MSE function. That’s it.        - 
Brett August 24, 2019 at 4:07 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498209 "Direct link to this comment")I’m confused about why you chose MSE for a classification problem. I was trying to use this tutorial to discern the differences between a classification and function approximation implementation, and the use of MSE for classification really threw me off. I know that it technically works, but it’s probably good to mention that it’s not ideal. It would have been nice to get exposure to taking the derivative of a different loss function, so that someone who is new to back-propagation will start to grasp how different functions change the derivative, etc. Otherwise, the code is understandable and could be modified slightly to make a good tutorial for function approximation.            - 
Brett August 24, 2019 at 4:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498211 "Direct link to this comment")Just to clarify what I’m saying, and to answer Pipo’s question, this implementation is already using MSE. The derivative of MSE with respect to the output is: (output – expected). The fact that you’re multiplying this by the transfer derivative just means that you’re passing the MSE back through the activation of the output node. So if you want the code to work for function approximation, you simply don’t multiply by the transfer derivative. However, if you want classification to work better, you could use the derivative of a different loss function with respect to the output and predicted, and multiply that by your transfer derivative.                - 
[Jason Brownlee](https://machinelearningmastery.com) August 24, 2019 at 7:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498252 "Direct link to this comment")Agreed!                - 
Marcel June 7, 2021 at 6:51 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-612491 "Direct link to this comment")Hello Brett,could you please point out where exactly the MSE loss is calculated? And where you would put the Cross entropy loss? Could you please demonstrate this with a short code example based on the tutorial ? I would be very pleased. Thank you in advance.            - 
[Jason Brownlee](https://machinelearningmastery.com) August 24, 2019 at 7:58 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498250 "Direct link to this comment")Yes, it was what we did in the 90s. Cross entropy would be preferred today, I agree.I really need to do a series on coding neural nets from scratch to really dig into this. Thanks for the kick!                - 
Brett August 24, 2019 at 9:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498260 "Direct link to this comment")Thanks for the reply. I just found this post on finding the derivative of cross entropy, and it turns out that you can do a really nice simplification of the math to basically get (output – expected) or (expected – output) for your implementation, when combining the cross entropy derivative and sigmoid derivative. So I’m pretty sure that if you simply stop multiplying by the transfer derivative to get your output error, you should see a big increase in performance of the algorithm. Worth a try at least. Here is the link, with the conclusion I mentioned at the very end:[https://peterroelants.github.io/posts/cross-entropy-logistic/](https://peterroelants.github.io/posts/cross-entropy-logistic/)                - 
[Jason Brownlee](https://machinelearningmastery.com) August 25, 2019 at 6:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498402 "Direct link to this comment")Thanks for sharing.153. 
muhammad December 21, 2018 at 8:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-459421 "Direct link to this comment")hi, thanks for this code.\
 I’m trying to understand why are u adding on the update weights, shouldnt be\
 wi←wi−η∂E/∂wi like this?154. 
Sangeeth January 20, 2019 at 4:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464254 "Direct link to this comment")Hi, This website provides a good introduction for almost all topics in machine learning. Thanks for your work. In backpropagation, the error at each neuron is the product of\
 1. Change in error w.r.t y\_out\
 2. Change in y\_out w.r.t y\
 3. Change in y w.r.t weight.Could you please tell how you just multiplied 1 and 2 in backward\_propagate\_error (from the last layer) and then used 3 in update\_weights (from the first layer). Should we not do all steps in backward\_propagate\_error and then use it to update\_weights?.    - 
[Jason Brownlee](https://machinelearningmastery.com) January 20, 2019 at 5:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464272 "Direct link to this comment")I show exactly how in the above tutorial.        - 
sangeeth January 20, 2019 at 5:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464274 "Direct link to this comment")Sorry, I just realized what I said is same as what you did.About the error, should we not use 2\*error (derivative of MSE)?            - 
[Jason Brownlee](https://machinelearningmastery.com) January 21, 2019 at 5:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464376 "Direct link to this comment")No, we calculate the derivative of the error against the non linear activation function, not the derivative of the loss function itself.                - 
sangeeth January 21, 2019 at 12:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464424 "Direct link to this comment")Ok. I got it. Thanks,I think this is online learning using SGD. Would you have an implementation for offline learning using mini Batch Gradient descent?                - 
[Jason Brownlee](https://machinelearningmastery.com) January 22, 2019 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464498 "Direct link to this comment")Correct, you can modify the above example to use batch or mini-batch gradient descent.                - 
sangeeth January 22, 2019 at 1:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464478 "Direct link to this comment")Is the sum\_error variable same as loss in model.fit output?..I get different loss values when testing the same datasets on model.fit and your model. Could you tell me why this is?                - 
[Jason Brownlee](https://machinelearningmastery.com) January 22, 2019 at 6:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464513 "Direct link to this comment")Yes, see this post:\
 [https://machinelearningmastery.com/randomness-in-machine-learning/](https://machinelearningmastery.com/randomness-in-machine-learning/)            - 
Brett August 24, 2019 at 4:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-498213 "Direct link to this comment")The loss function used in this tutorial is: (1/2)(out – expect)\^2. The derivative of which with respect to the output is: (out – expect) \* 1, or simply (out – expected). This is then multiplied by the transfer derivative, because the error is being passed backward via the chain rule. You always have to take the derivative with respect to the loss function itself first. I hope this clears up any confusion.155. 
Gary January 22, 2019 at 8:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464529 "Direct link to this comment")Hi Jason.In the “full” seeds example you call user defined function evaluate\_algorithm(). However, the “heavy lifting” inside it is performed by the function algorithm(). That function looks like it’s a part of some standard Python library, but I can’t find it in any reference. Also you don’t comment at all at its use. What’s the deal here?Thank you,Gary    - 
[Jason Brownlee](https://machinelearningmastery.com) January 22, 2019 at 11:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464545 "Direct link to this comment")The “algorithm” is a reference to a function that is passed in as an argument.156. 
Gary January 22, 2019 at 1:24 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-464556 "Direct link to this comment")Yes, thank you, I already realized that.Regards,Gary157. 
sangeeth January 28, 2019 at 8:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-465278 "Direct link to this comment")Hi,For online machine learning, should we perform epochs?. Should not we update the model based only on the present time input and then predict the next time step. If we do epochs that means the model is getting updated for the whole data set up to the present time. Am I correct?. Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) January 28, 2019 at 11:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-465289 "Direct link to this comment")It depends on the problem and the data. Yes, if often makes sense to update the model with new data and with a little of the old data.Note, online gradient descent does not have to be used for online learning.158. 
kmillen February 12, 2019 at 10:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-467953 "Direct link to this comment")Good afternoon Jason. I have thoroughly enjoyed this solution both in Python and my conversion to C#. I guess for all the learning I’ve gleaned, one thing still seems to be a mystery to me. What exactly are the five scores telling me? Do they annotate how well the data fits a curve for each fold?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 12, 2019 at 1:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-467967 "Direct link to this comment")Correct. The mean of the scores is our estimate of the model’s performance when making predictions on unseen data.        - 
kmillen February 26, 2019 at 8:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-470677 "Direct link to this comment")Thank you.159. 
MathewP February 20, 2019 at 3:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469383 "Direct link to this comment")I think there is a mistake update\_weights function.\
 inputs \= row\[:-1\]\
 If we have, say 2 inputs and 1 neuron in hidden layer then only one weight is going to be updated, which is clearly wrong. Correct me if I am wrong. The code works fine just taking row as inputs.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 20, 2019 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469418 "Direct link to this comment")I don’t follow the possible issue, can you please elaborate?        - 
Romel Rudon October 20, 2019 at 11:25 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-506506 "Direct link to this comment")The issue is that the ‘row’ list should represent the outputs from the preceding layer (counting in the direction from input layer to output layer). having row\[:-1\] seems to exclude the very last output from the preceding layer, which doesn’t seem to be warranted in this case.            - 
Romel Rudon October 21, 2019 at 2:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-506530 "Direct link to this comment")I see now why a few people (including myself) were thrown off by this line. The last element of the row list ( i.e. row\[-1\]) is not an actual part of the input data, but the label or the ‘correct answer’ of the input data, which is why it’s left out. Cheers.                - 
[Jason Brownlee](https://machinelearningmastery.com) October 21, 2019 at 6:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-506577 "Direct link to this comment")Yes.160. 
Venkat February 22, 2019 at 4:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469794 "Direct link to this comment")Hi Jason Brownlee, back propagation implementation really excellent ,because of without using any predefined library just use functions list, set, and dictionary. I need a suggestion how to write a code for implement activation function like a sigmoidal at hidden layer neurons and a tangent at output neurons. could u help me.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 22, 2019 at 6:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469830 "Direct link to this comment")Yes, you can use the above as a starting point.More on tanh here:\
 [https://en.wikipedia.org/wiki/Hyperbolic\_function](https://en.wikipedia.org/wiki/Hyperbolic_function)161. 
Venkat February 22, 2019 at 4:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469875 "Direct link to this comment")Hi Jason Brownlee, yes , iam not asking how to write code for implementation of tanh, sigmoidial . My request is how to modified code in forward\_propagate function to implement suppose x is a activation at hidden layer and y is another activation function at output layer. 12345678910def forward\_propagate ( network ,  row ) : inputs \= row  for  layer in  network :new\_inputs  \=  \[ \]for  neuron in  layer :activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )neuron \[ 'output' \]  \=  transfer ( activation )new\_inputs . append ( neuron \[ 'output' \] ) inputs \= new\_inputs     return inputsin the above code u r calling transfer function for the hidden neurons and also output neurons . I request u to give suggestion to call different activation functions for hidden and output neurons.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 23, 2019 at 6:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-469954 "Direct link to this comment")Change the code in the activation function itself.Does that help?162. 
[Danh Nguyen](https://danh-was-here.netlify.com/) February 24, 2019 at 12:09 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-470270 "Direct link to this comment")Example is great! The totally clean CSV wheat seed dataset is here: [https://raw.githubusercontent.com/NguyenDa18/MachineLearning\_HW3/master/wheat-seeds.csv](https://raw.githubusercontent.com/NguyenDa18/MachineLearning_HW3/master/wheat-seeds.csv)I tried Jason’s link\
 [https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv) \
 and the UCI Repo link and the CSVs still had double commas and so we got the str\_column\_to\_float errorAnyway, posting this here so others won’t run into the same problem I did! Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) February 25, 2019 at 6:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-470428 "Direct link to this comment")Thanks for sharing.163. 
vartika sharma February 27, 2019 at 3:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-470846 "Direct link to this comment")Hey Jason,\
 While I am ruuning the following code, I am getting this error\
 >>> scores\=evaluate\_algorithm(dataset,back\_propagation,n\_folds,l\_rate,n\_epoch,n\_hidden)\
 Traceback (most recent call last):\
 File “”, line 1, in\
 File “”, line 13, in evaluate\_algorithm\
 File “”, line 5, in back\_propagation\
 File “”, line 6, in train\_network\
 TypeError: list indices must be integers, not str    - 
[Jason Brownlee](https://machinelearningmastery.com) February 27, 2019 at 7:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-470896 "Direct link to this comment")Perhaps try saving code to a file and run from the command line, here’s how:\
 [https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line](https://machinelearningmastery.com/faq/single-faq/how-do-i-run-a-script-from-the-command-line)164. 
Andy March 5, 2019 at 12:51 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-472429 "Direct link to this comment")Hello Jason,I would like to ask, can you make the data split between training data and test data, instead of using k folds variation, I would like to get some insight in this, thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) March 5, 2019 at 2:22 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-472454 "Direct link to this comment")Yes, I show how here:\
 [https://machinelearningmastery.com/implement-resampling-methods-scratch-python/](https://machinelearningmastery.com/implement-resampling-methods-scratch-python/)165. 
Andy March 5, 2019 at 6:33 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-472518 "Direct link to this comment")Hello Jason, it’s me again.I would like to ask another question, how do you predict using this trained network ?\
 Lets say I have 100 data, and I split the training and test by 70:30 ratio. I’ve trained the network using 70 data, how do I predict the rest (30 data) ?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 6, 2019 at 7:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-472690 "Direct link to this comment")You can fit one final model on all data, then use it to make predictions (see the predict section).Remember, this is an example for learning only. If you want a model for your data in practice, I recommend using a robust neural network library like Keras:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)166. 
Dini M March 11, 2019 at 12:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-473477 "Direct link to this comment")# Convert string column to integer\
 def str\_column\_to\_int(dataset, column):\
 class\_values \= \[row\[column\] for row in dataset\]\
 unique \= set(class\_values)\
 lookup \= dict()\
 for i, value in enumerate(unique):\
 lookup\[value\] \= i\
 for row in dataset:\
 row\[column\] \= lookup\[row\[column\]\]\
 return lookupI got the error “class\_values \= \[row\[column\] for row in dataset\]”\
 IndexError: list index out of range    - 
[Jason Brownlee](https://machinelearningmastery.com) March 11, 2019 at 6:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-473520 "Direct link to this comment")Sorry to hear that, I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)167. 
Dini M March 11, 2019 at 12:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-473478 "Direct link to this comment")I trying your code example and seeds\_dataset.csv168. 
giuseppe March 15, 2019 at 7:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-474402 "Direct link to this comment")Hi thanks for the code is amazing,\
 I’ve included it in a class for a project, I’ve modified it so I can decide how many neurons put in each layer because, but I have a question: during the train process using for example 4 neurons in the first layer and 3 in the second one I get nice result, around 85% / 92 %. At this point I save all the weights of the neurons and I call another function that just load the weight that I’ve saved(skipping in this way the traning process) and using all the dataset(the same I’ve used for train the network) as test set it gives me a really bad score, most of the time is around 30%. I’m using the “IrisDataTrain” and what I’ve noticed is that the networks fails to recognise one of the 3 classes. Do you have any suggestion about what could be? Thanks 🙂    - 
[Jason Brownlee](https://machinelearningmastery.com) March 15, 2019 at 2:25 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-474457 "Direct link to this comment")Perhaps the weights are not being loaded correctly?        - 
giuseppe March 28, 2019 at 8:11 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476947 "Direct link to this comment")Hi sorry for the late replay, actually the problem of save and loading the nn is not that important so maybe i’ll try to solve it later. At the moment the problem is that i should reach the 99. % on the “IrisDataTrain” set. What i’ve noticed is that the accurancy can change a lot repeating the same training process with the same configuration. In order to get a better result I’ve tryied to repeat the same traning process different times with the same configuration, I’ve choosen the configuration that give me the best result in mean and variance. Now in order to improve the accurancy I’ve modified the code so that I can connect easily the output of a nn with the input of another one so that I can create a cascade of neural networks connected in different ways. At this point i’m stucked at 96% in mean. To improve the accurancy I’ve implemented the relu activation function (but i’m not sure it’s correctly implemented) and adam optimizer (but it doesn’t work at all).\
 I’ll link the code on pastbin (I don’t know if there is any better way to do that) in particular what I’ve done is just insert everything in a class and modified:\
 1. the initialization function so that I can chose the number of neurons in each layer\
 2. the back\_propagation\_error function trying to add the relu and adam optimizer\
 3. the update weights function trying to implement adam optimizer (it doesnt’ work at all)\
 In the code I’m going to share I’ve just removed many parts just for a readability reason, after cleaning it I will send it to you if you want. Sorry for the long message and thanks for you help 🙂[https://pastebin.com/RxxuVaCD](https://pastebin.com/RxxuVaCD)            - 
[Jason Brownlee](https://machinelearningmastery.com) March 29, 2019 at 8:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477097 "Direct link to this comment")Nice work!It might be time to graduate to Keras where everything is implemented for you and you can just use it directly and focus on tuning the model.169. 
MLnovice March 21, 2019 at 10:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-475784 "Direct link to this comment")Hello sir,\
 I am playing with your code and I am trying to figure out this error:line 185, in\
 str\_column\_to\_float(dataset, i)\
 line 21, in str\_column\_to\_float\
 row\[column\] \= float(row\[column\].strip())\
 ValueError: could not convert string to float: Do you have any insides of why this is happening?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 22, 2019 at 8:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-475859 "Direct link to this comment")Sorry to hear that, I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)170. 
Matty March 25, 2019 at 9:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476234 "Direct link to this comment")Thank you for the post Jason.Reading this post, it seems to me that I can split the process of back propagation in large networks into multiple steps. Am I right? I have a large network that my current GPU runs out of memory when I try to train it. I was wondering if I can split my network into two sub-networks, and first calculate the updates for the deeper part(that has the ground truth outputs) and obtain the error that should be passed to the other sub-network. Then, use the provided error to calculate the updates for the second sub-network as well. Do you think it’s possible? Do you have any suggestion (or source that can be helpful) for implementing this back propagation with existing tensorflow or pytorch builtin functions?Thanks.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 25, 2019 at 2:17 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476263 "Direct link to this comment")Yes, by node or by layer.It might be possible, but also a massive pain.It might be cheaper (in time/money) to rent an AWS EC2 instance with more GPU RAM for a few hours?        - 
Matty March 26, 2019 at 1:03 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476414 "Direct link to this comment")Thanks, Jason. I think I found an easy way to split the back-propagation in tensorflow. We can define two separate optimizations with different trainable variable lists. Something similar to: self.optim\_last\_layers \= tf.train.AdamOptimizer(lr, beta1\=beta1) \\
 .minimize(loss, var\_list\=vars\_of\_last\_layers)  self.optim\_first\_layers \= tf.train.AdamOptimizer(lr, beta1\=beta1) \\
 .minimize(loss, var\_list\=vars\_of\_first\_layers) And in each iteration, we can call the optimizations separately. I did a small sanity check with a two-layer network, and it seems both the two-step optimization and the one-step optimization with all the trainable parameters results in the same updates.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 26, 2019 at 2:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476427 "Direct link to this comment")Glad to hear it.171. 
Novia Puspitasari March 28, 2019 at 12:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-476723 "Direct link to this comment")thankyou so much jason, for your post about it\
 i have some problem in “‘float’ object has no attribute ‘append'”Traceback (most recent call last):\
 File “backprop.py”, line 200, in\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 File “backprop.py”, line 80, in evaluate\_algorithm\
 predicted \= algorithm(train\_set, test\_set, \*args)\
 File “backprop.py”, line 172, in back\_propagation\
 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 File “backprop.py”, line 150, in train\_network\
 backward\_propagate\_error(network, expected)\
 File “backprop.py”, line 123, in backward\_propagate\_error\
 error.append(error)\
 AttributeError: ‘float’ object has no attribute ‘append’do you have a solve for that ? thankyou172. 
Kevin March 29, 2019 at 1:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477009 "Direct link to this comment")Hi there Jason, I would like to ask if I wanted to generate random weights and bias with range -1 to 1, how to do it ? Since I already import random or from uniform import random, and it said AttributeError: ‘builtin\_function\_or\_method’. Thank you very much !    - 
[Jason Brownlee](https://machinelearningmastery.com) March 29, 2019 at 8:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477109 "Direct link to this comment")Good question, this post shows how:\
 [https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/](https://machinelearningmastery.com/how-to-generate-random-numbers-in-python/)You then scale them to any range you want: result \= min \+ value \* range        - 
Kevin March 29, 2019 at 4:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477201 "Direct link to this comment")Thanks for your reply ! Awesome guide, I really grateful for it. Once again, thanks a lot Jason.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 30, 2019 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477323 "Direct link to this comment")You’re welcome.173. 
wancong zhang March 30, 2019 at 12:45 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477382 "Direct link to this comment")Hi Jason, very cool tutorial.I notice that your neural network only has 3 layers.If I change your “initialize network” method to initialize multiple hidden layers with arbitrary width, will your program still work? In other words does your algorithm generalize to deeper networks?Thanks.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 31, 2019 at 9:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-477547 "Direct link to this comment")No idea – it is for educational purposes only, try it and see.174. 
[manoj](http://none) April 19, 2019 at 7:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481210 "Direct link to this comment")Hi Jason!Its really a helpful post, that you very much. I wanted to see the plots of training error and testing error. (like how they finally converged by epochs by epochs). What would be the easiest way to plot those training and testing graphsregards\
 Manoj Goli    - 
[Jason Brownlee](https://machinelearningmastery.com) April 19, 2019 at 3:03 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481254 "Direct link to this comment")I’d recommend using a library like Keras which provides the history directly and ready to plot:\
 [https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)175. 
Danial April 20, 2019 at 5:46 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481476 "Direct link to this comment")Hi jason.\
 My question is how I can see my CNN code is using BP framework?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 21, 2019 at 8:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481564 "Direct link to this comment")You can save the model weights to a file.Is that what you mean?        - 
Danial April 21, 2019 at 1:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481626 "Direct link to this comment")Yes. How I can see model weights? How cnn use BP framework if it is not shown in code?\
 for i in range(len(test)):\
 # Forecast the data\
 test\_X, test\_y \= test\[i, 0:-1\], test\[i, -1\]\
 X\_ \= test\_X.reshape(1, 28, 28, 1)\
 predict \= model.predict(X\_, batch\_size\=1)\
 predict \= predict\[0, 0\] # Replacing value in test scaled with the predicted value.\
 test\_prediction \= \[predict\] \+ test\_prediction\
 if len(test\_prediction) > sequence\_length\+1:\
 test\_prediction \= test\_prediction\[:-1\]\
 if i\+1 sequence\_length\+1:\
 test\[i\+1\] \= test\_prediction\
 else:\
 test\[i\+1\] \= np.concatenate((test\_prediction, test\[i\+1, i\+1:\]), axis\=0) # Inverse transform\
 predict \= inverse\_transform(scaler, test\_X, predict)\
 # Inverse the features\
 predict \= inverse\_features(data\_set, predict, len(test)\+1-i) – maxVal\
 if predict \< 0:\
 predict \= 0\
 # Round the value\
 predict \= np.round(predict, 2)\
 # store forecast\
 expected \= data\_set\[len(train) \+ i \+ 1\]\
 predict\_data.append(predict )\
 real\_data.append(expected )\
 if expected !\= 0:\
 prediction.append(predict)\
 real.append(expected)            - 
[Jason Brownlee](https://machinelearningmastery.com) April 22, 2019 at 6:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481796 "Direct link to this comment")You can get the model weights from a Keras model by calling the get\_weights() function on a give layer.                - 
Danial April 22, 2019 at 6:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481813 "Direct link to this comment")Is it right that above code is using BP framework.? It’s part of CNN code that I sent                - 
Danial April 22, 2019 at 11:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481858 "Direct link to this comment")How keras uses BP framework.? If you have link kindly share it.                - 
[Jason Brownlee](https://machinelearningmastery.com) April 22, 2019 at 2:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-481889 "Direct link to this comment")You can get started with Keras here:\
 [https://machinelearningmastery.com/start-here/#deeplearning](https://machinelearningmastery.com/start-here/#deeplearning)176. 
Idris Shareef April 24, 2019 at 4:16 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-482358 "Direct link to this comment")Hello Jason , You’re the best teacher.    - 
[Jason Brownlee](https://machinelearningmastery.com) April 25, 2019 at 8:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-482456 "Direct link to this comment")Thanks.177. 
Abarni April 30, 2019 at 12:39 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-483373 "Direct link to this comment")Nice Post !Here is another very nice tutorial with step by step Mathematical explanation and full coding.[http://www.adeveloperdiary.com/data-science/machine-learning/understand-and-implement-the-](http://www.adeveloperdiary.com/data-science/machine-learning/understand-and-implement-the-) \
 backpropagation-algorithm-from-scratch-in-python/    - 
[Jason Brownlee](https://machinelearningmastery.com) April 30, 2019 at 2:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-483390 "Direct link to this comment")Thanks for sharing.178. 
Zahra May 6, 2019 at 4:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484227 "Direct link to this comment")Hello, I’m so confuse.\
 I try to run this code in command prompt. But, I use my dataset (not Wheat Seeds dataset).And why this happened? What’s wrong? What should I do? What should I change?\
 Please, help me!Traceback (most recent call last):\
 File “journal.py”, line 197, in\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 File “journal.py”, line 81, in evaluate\_algorithm\
 predicted \= algorithm(train\_set, test\_set, \*args)\
 File “journal.py”, line 173, in back\_propagation\
 train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 File “journal.py”, line 150, in train\_network\
 expected\[row\[-1\]\] \= 1\
 IndexError: list assignment index out of range    - 
[Jason Brownlee](https://machinelearningmastery.com) May 7, 2019 at 6:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484279 "Direct link to this comment")I cannot know, I can give you some advice on debugging your problem here:\
 [https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code](https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code)179. 
Zahra May 9, 2019 at 1:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484500 "Direct link to this comment")Hello, How to import my dataset using that codes? For example (in this codes), how to use my dataset (use excel file) in this codes.. How to import my dataset in this codes? Can you teach me more detail, please..# Test training backprop algorithm\
 seed(1)\
 dataset \= \[\[2.7810836,2.550537003,0\],\
 \[1.465489372,2.362125076,0\],\
 \[3.396561688,4.400293529,0\],\
 \[1.38807019,1.850220317,0\],\
 \[3.06407232,3.005305973,0\],\
 \[7.627531214,2.759262235,1\],\
 \[5.332441248,2.088626775,1\],\
 \[6.922596716,1.77106367,1\],\
 \[8.675418651,-0.242068655,1\],\
 \[7.673756466,3.508563011,1\]\]\
 n\_inputs \= len(dataset\[0\]) – 1\
 n\_outputs \= len(set(\[row\[-1\] for row in dataset\]))\
 network \= initialize\_network(n\_inputs, 2, n\_outputs)\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 for layer in network:\
 print(layer)    - 
[Jason Brownlee](https://machinelearningmastery.com) May 9, 2019 at 6:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484540 "Direct link to this comment")Perhaps you should start with Keras, it is much easier for beginners:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)        - 
Aziz Ahmad July 22, 2019 at 4:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-493696 "Direct link to this comment")Sir! I really apprecitae your work.            - 
[Jason Brownlee](https://machinelearningmastery.com) July 22, 2019 at 8:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-493722 "Direct link to this comment")Thanks.180. 
Nirmala May 10, 2019 at 5:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484641 "Direct link to this comment")I got an error called-> IndexError: list assignment index out of range.but I m using python 3 itself.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 10, 2019 at 8:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484672 "Direct link to this comment")Sorry to hear that, I have some suggestions here:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)181. 
Ido Berenbaum May 11, 2019 at 9:39 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484917 "Direct link to this comment")Hi Jason,\
 thanks for the great tutorial, I learned a lot from it.\
 There is one thing I didn’t really understand though,\
 when you update the weights you add to the weight the calculated change that is needed.\
 but’from what I read in other sites like wikipedia, the change to the weight needs to be multiplied by -1 and then added to ensure it\
 changes the weight in the opposite direction of the gradient and so getting it closer to the local minimum.\
 like muhammad said in December 21, 2018:\
 “hi, thanks for this code.\
 I’m trying to understand why are u adding on the update weights, shouldnt be\
 wi←wi−η∂E/∂wi like this?”and I tried to change line 141 to: neuron\[‘weights’\]\[j\] -\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 basically doing -\= and not \+\= but it just made the sum error of the network to increase after each epoch.so, I will be grateful if you could explain to me why are you adding and not subtracting.thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) May 12, 2019 at 6:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-484973 "Direct link to this comment")There are many ways to implement the algorithm description.This implementation is based on the description in “neural smithing”:\
 [https://amzn.to/2pW6hjI](https://amzn.to/2pW6hjI)    - 
cocoa July 16, 2019 at 8:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-493036 "Direct link to this comment")Jason seem to use mean square error as loss. partial derivative of loss should be (output-expected). In his “backward” function, he did (expected-output). That’s why he came up with “\+\=” not “-\=”        - 
[Jason Brownlee](https://machinelearningmastery.com) July 16, 2019 at 2:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-493053 "Direct link to this comment")Nice explanation.182. 
Nirmala May 16, 2019 at 4:23 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-485613 "Direct link to this comment")In training code and testing code I want to link the onther dataset .txt file but it will not work.please can u send a code for that..    - 
[Jason Brownlee](https://machinelearningmastery.com) May 17, 2019 at 5:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-485675 "Direct link to this comment")Sorry, I don’t have the capacity to develop custom code for you.183. 
Arthur May 22, 2019 at 4:03 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486255 "Direct link to this comment")Hi Jason, first of all thank you very much for this post, I’m learning ML at the moment, and writing a neural network with backpropagation in C# to help the process.When using the wheat seeds dataset, and the same network layout as you suggest, I get very similar results to yours in terms of accuracy.When I scale the network however to have 2 (or more) hidden layers of 5 neurons, I sometimes get exploding gradients (and NaN output results because of it). Is this something you’d expect given the dataset? ie, doesn’t this dataset allow for much more than 1 hidden layer for some reason. (Some context: I normalize the data just like you do, and I use the same dataset as you do, the reason I test with more than 1 layer, is just out of curiosity, and to see whether the accuracy improved – it doesnt)I’m trying to understand why it happens with this particular data, or whether my implementation fails somehow. Note that I do get good results most of the time, but with a certain weight initialization the exploding gradients can happen.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 23, 2019 at 5:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486328 "Direct link to this comment")Perhaps try scaling the data prior to modeling, see this post:\
 [https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)184. 
Zahra Nabila May 27, 2019 at 4:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486790 "Direct link to this comment")Hello, I have problem. Why output must be integers, not float (decimal)? Specially in Train.. How to change output data type to float?TypeError Traceback (most recent call last) in train\_network(network, train, l\_rate, n\_epoch, n\_outputs)\
 93 outputs \= forward\_propagate(network, row)\
 94 expected \= \[0 for i in range(n\_outputs)\]\
 —> 95 expected\[row\[-1\]\] \= 1\
 96 sum\_error \+\= sum(\[(expected\[i\]-outputs\[i\])\*\*2 for i in range(len(expected))\])\
 97 backward\_propagate\_error(network, expected)TypeError: list indices must be integers or slices, not numpy.float64    - 
[Jason Brownlee](https://machinelearningmastery.com) May 28, 2019 at 8:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486849 "Direct link to this comment")The example is classification, you can change it to regression if you wish.Perhaps this tutorial will be more helpful:\
 [https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)        - 
Zahra May 28, 2019 at 1:09 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486885 "Direct link to this comment")I want to ask, how to display execution time (running rime) in Train code?            - 
[Jason Brownlee](https://machinelearningmastery.com) May 28, 2019 at 2:44 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-486891 "Direct link to this comment")Sorry, I don’t have an example of calculating clock time for code examples.185. 
Jeny June 3, 2019 at 10:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-487711 "Direct link to this comment")”’\
 Calculate the output of a recurrent neural network with tanh activation\
 and a linear layer on top\
 Params:\
 x: input matrix \[n\_timesteps \* n\_samples \* 2\]\
 w: non-recurrent weights\
 r: recurrent weights\
 b: biases\
 wo: output-layer weights\
 bo: output-layer biases\
 Returns:\
 h: matrix of activations (n\_timesteps, n\_samples, n\_hiddens)\
 o: final predictions\
 ”’def forward\_path(x, w, r, b, wo, bo):\
 h \= np.empty(\[t\_max, n, w.shape\[0\]\], dtype\=np.float32) # storage for the hidden activations\
 for t in range(t\_max):\
 z \= np.dot(x\[t\], w.T) \+ b\
 if t > 0:\
 z \+\= np.dot(h\[t-1\], r.T)\
 h\[t\] \= np.tanh(z)\
 o \= np.dot(h\[-1\], wo.T) \+ bo\
 return h, odef backward\_path(x, h, w, b, r, wo, bo, o, y):\
 n, t\_max, \_ \= x.shape\
 dw \= np.zeros\_like(w)\
 db \= np.zeros\_like(b)\
 dr \= np.zeros\_like(r)\
 dwo \= 0\
 dbo \= 0return dw, dr, db, dwo, dbodef loss(w, r, b, wo, bo, x, y):\
 \_, o \= forward\_path(x, w, r, b, wo, bo)\
 err \= 0.5\*np.sum(np.square(o-y))\
 return errCan you help me implement the backpropagation? Please.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 3, 2019 at 2:34 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-487728 "Direct link to this comment")Sorry, I don’t have the capacity to debug your code, perhaps try posting to stackoverflow?186. 
Zahra Nabila Izdihar June 10, 2019 at 11:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-488476 "Direct link to this comment")Hello…\
 How to display “predicted” value in your code?Because I need to display the predicted or forecast value..Thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) June 11, 2019 at 7:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-488516 "Direct link to this comment")The forward\_propagate() function makes a prediction.        - 
Zahra Nabila Izdihar June 13, 2019 at 2:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-488715 "Direct link to this comment")I got it. So, “output” (in ForwardPropagation code)\= prediction result? But, I don’t understand How to determine the weights in forward propagation? What is the formula?Thank you            - 
[Jason Brownlee](https://machinelearningmastery.com) June 13, 2019 at 6:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-488741 "Direct link to this comment")What do you mean exactly?The weights are learned during training.                - 
Zahra June 16, 2019 at 4:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-489059 "Direct link to this comment")Did you mean that “output” (in forward propagatation) is predicted result?                - 
[Jason Brownlee](https://machinelearningmastery.com) June 16, 2019 at 7:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-489078 "Direct link to this comment")Yes.Perhaps this is too advanced. I recommend starting with Keras instead:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)187. 
Leo July 2, 2019 at 11:25 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-491304 "Direct link to this comment")It is crazy that nobody complain the readability of your codes. Thanks anyway    - 
[Jason Brownlee](https://machinelearningmastery.com) July 3, 2019 at 8:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-491360 "Direct link to this comment")Sorry that you think that the code is not readable. I thought it was very readable.What do you believe the problem is exactly?188. 
Femi July 14, 2019 at 12:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-492782 "Direct link to this comment")Good day Dr. I am new to machine learning but have interest in it. My question is this, can i use the Python 2.7 in miniconda to implement your samples?    - 
[Jason Brownlee](https://machinelearningmastery.com) July 14, 2019 at 8:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-492826 "Direct link to this comment")Yes.189. 
Ravi July 19, 2019 at 9:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-493477 "Direct link to this comment")Hi Dr. JasonI have developed and trained a neural network (3 layers: 1 input, 1 hidden and 1 output) for following situation(The code was written step by step, as i do not want to use a tool without understanding the computations) Data set (40 input patterns): Input: 40 samples 5 elements\
 Output: 40 samples 1 element\
 number of neurons (Input \= 5; hidden \= 5; output \= 1)Using the delta rule with backpropagation algorithm, i was able to achieve error \= 9.39E-06 for 1000 iterationsMy final “input to hidden layer” weight matrix size is 200 x 5 (as i have 40 samples x 5 input neurons and 5 hidden neurons) “hidden to output layer” weight matrix size is 200 x 1 (as i have 40 samples x 5 hidden neurons and 1 output neuron)Now my question is for a given test sample having 5 elements (input is 1 sample 5 elements), i need to run feed-forward computation to get a single element output.For running this which weights i need to select in “input to hidden layer” and “hidden to output layer” from the trained set??I have 200 x 5 and 200 x 1 weight matrices; but i require only 5 x 5 and 5 x 1 weight matrices for testing.Kindly let me know if i am missing something here?Thanks in advanceRavi190. 
Chrissie Li July 25, 2019 at 9:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494218 "Direct link to this comment")hi, how can i download the code? copy the tip’s script?    - 
[Jason Brownlee](https://machinelearningmastery.com) July 26, 2019 at 8:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494265 "Direct link to this comment")I show you how right here:\
 [https://machinelearningmastery.com/faq/single-faq/how-do-i-copy-code-from-a-tutorial](https://machinelearningmastery.com/faq/single-faq/how-do-i-copy-code-from-a-tutorial)191. 
Femi July 25, 2019 at 9:39 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494219 "Direct link to this comment")Prof Sir, pls i have been finding it difficult to implement your samples when i can not prepare environment that accept all the command. Pls help    - 
[Jason Brownlee](https://machinelearningmastery.com) July 26, 2019 at 8:23 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494266 "Direct link to this comment")This tutorial will show you how to setup your environment:\
 [https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)192. 
Femi July 31, 2019 at 12:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494827 "Direct link to this comment")Sir, I guessed you use scipy environment. am i right?    - 
[Jason Brownlee](https://machinelearningmastery.com) July 31, 2019 at 6:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-494868 "Direct link to this comment")For this tutorial, a simple Python environment is enough.193. 
Majed August 4, 2019 at 8:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-495423 "Direct link to this comment")I wrote a neural network that consists of three layers as follows:\[ 4 input neurones – 5 hidden neurones – 3 output neurones\]. first, I standerdized the data using the z-score. The accuracy of my model exceeded 67. Note: I didn’t use the regularisation terms yet.\
 here is my implementation of both feedforward and back prop .. 1234567891011121314while  iteration  \<  number\_of\_iterations :z2  \=  np . dot ( x ,  w1 )a2  \=  sigmoid ( z2 )z3  \=  np . dot ( a2 ,  w2 )a3  \=  sigmoid ( z3 )  # this represents the output of the networkerror  \=  loss\_function ( a3 ,  label\_matrix )delta\_3  \=  np . multiply ( - ( label\_matrix - a3 ) ,  sigmoid\_prime ( z3 ) )dJW2  \=  np . dot ( a2 . transpose ( ) ,  delta\_3 )delta\_2  \=  np . dot ( delta\_3 ,  w2 . transpose ( ) )  \*  sigmoid\_prime ( z2 )dJW1  \=  x . transpose ( )  @  delta\_2w2  \=  w2  -  ( learning\_rate \*  dJW2 )w1  \=  w1  -  ( learning\_rate \*  dJW1 )iteration  \=  iteration  \+  1  # update the counter :')return  w1 ,  w2Thanks …    - 
Majed August 4, 2019 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-495424 "Direct link to this comment")The data set that I worked on is the Iris data set    - 
[Jason Brownlee](https://machinelearningmastery.com) August 5, 2019 at 6:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-495505 "Direct link to this comment")Well done!194. 
Ekundayo August 6, 2019 at 9:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-495742 "Direct link to this comment")Ca I still get an help this one time? Pls sir, my project is to use 14 features extractions for plant leave classification. I need to recognize one leaf out of 36 leaves all with 14 features. Sir can I use your code to achieve this?\
 Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) August 7, 2019 at 7:52 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-495795 "Direct link to this comment")I would recommend this tutorial, e.g. transfer learning:\
 [https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)195. 
Mohammed August 14, 2019 at 12:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496710 "Direct link to this comment")Hi Dr. JasonThank you for this post, it is really very helpful.I have one question about backpropagation in unsupervised model, e.g. extract features.\
 Is it possible to apply this code for it, and only replaces loss function of unsupervised model by the loss function of supervised?Regards    - 
[Jason Brownlee](https://machinelearningmastery.com) August 14, 2019 at 2:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496724 "Direct link to this comment")Backpropagation is for supervised learning, not unsupervised learning.        - 
Mohammed August 15, 2019 at 5:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496856 "Direct link to this comment")Oh! many thanks,So, can help me what is the way for learning parameters in unsupervised approach.\
 if i need to extract the features from data as low dimension nested of data with large dimension.            - 
[Jason Brownlee](https://machinelearningmastery.com) August 16, 2019 at 7:47 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496924 "Direct link to this comment")There are specialized techniques for unsupervised learning neural nets, perhaps start with the SOM:\
 [http://cleveralgorithms.com/nature-inspired/neural/som.html](http://cleveralgorithms.com/nature-inspired/neural/som.html)196. 
Mohammed August 15, 2019 at 5:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496857 "Direct link to this comment")such as Unsupervised feature learning with Sparse Filtering!    - 
[Jason Brownlee](https://machinelearningmastery.com) August 16, 2019 at 7:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496925 "Direct link to this comment")Sorry, I don’t have a tutorial on that topic, perhaps in the future.197. 
Mohammed August 16, 2019 at 11:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496966 "Direct link to this comment")Thank you so much Dr. Jason.    - 
[Jason Brownlee](https://machinelearningmastery.com) August 16, 2019 at 2:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-496986 "Direct link to this comment")You’re welcome.198. 
Cherinet Mores August 20, 2019 at 4:44 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497496 "Direct link to this comment")Jason Brownlee\
 Thank you for your continues help\
 Here I have one questions,\
 In case, if i want to solve the regression problem (Meaning, if I have 3 real value outputs from the input parameters) which part of the code should be modified and How? Thank you very much    - 
[Steven Pauly](http://www.slimstock.be) August 20, 2019 at 9:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497551 "Direct link to this comment")Hi Cherinet, I’ve changed the n\_outputs to 1 and the function train\_network, I’ve changed the below. I’ve increased the n\_epoch to a lot higher, because else it will give you the average. Be sure to normalize your input & output values, though.12345678910def train\_network ( network ,  train ,  l\_rate ,  n\_epoch ,  n\_outputs ) :for  epoch in  range ( n\_epoch ) : sum\_error \= 0for  row in  train :outputs  \=  forward\_propagate ( network ,  row )expected  \=  \[ row \[ - 1 \]  for  i  in  range ( n\_outputs ) \]sum\_error  \=  sum ( \[ ( expected \[ i \] - outputs \[ i \] ) \* \* 2  for  i  in  range ( len ( expected ) ) \] )backward\_propagate\_error ( network ,  expected )update\_weights ( network ,  row ,  l\_rate )print ( '>epoch\=%d, lrate\=%.3f, error\=%.3f'  %  ( epoch ,  l\_rate ,  sum\_error ) )        - 
[Jason Brownlee](https://machinelearningmastery.com) August 21, 2019 at 6:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497623 "Direct link to this comment")Thanks for sharing.        - 
Cherinet Mores August 21, 2019 at 8:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497648 "Direct link to this comment")Dear Steven Pauly thank you very much for your help.        - 
Charles September 17, 2019 at 1:01 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-501841 "Direct link to this comment")Hi, By normalizing input and output, do you mean modifying the forward\_propogate method like this?12345678910111213141516171819# Forward propagate input to a network outputdef forward\_propagate\_regression ( network ,  row ) : inputs \= rownew\_inputs  \=  \[ \] # gets the 1st layer, applies sigmoid activationhiddenlayer  \=  network \[ 0 \]for  neuron in  hiddenlayer :activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )neuron \[ 'output' \]  \=  transfer ( activation )new\_inputs . append ( neuron \[ 'output' \] )inputs  \=  new \_inputs # gets the last layer, applies linear activationoutputlayer  \=  network \[ - 1 \]for  neuron in  outputlayer :activation  \=  activate ( neuron \[ 'weights' \] ,  inputs )neuron \[ 'output' \]  \=  activationnew\_inputs . append ( neuron \[ 'output' \] ) inputs \= new\_inputs return inputs    - 
[Jason Brownlee](https://machinelearningmastery.com) August 21, 2019 at 6:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497611 "Direct link to this comment")Change the output to be a linear activation and the loss function to mse.199. 
[Steven Pauly](http://www.slimstock.be) August 20, 2019 at 9:56 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497550 "Direct link to this comment")Well Done, Jason! Great stuff!!!    - 
[Jason Brownlee](https://machinelearningmastery.com) August 21, 2019 at 6:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-497622 "Direct link to this comment")Thanks, I’m glad it helped.200. 
George Shannon September 25, 2019 at 12:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-502840 "Direct link to this comment")Dear Dr. Brownlee:Do you have an example of doing the same thing but with backprop using momentum?George    - 
[Jason Brownlee](https://machinelearningmastery.com) September 25, 2019 at 5:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-502880 "Direct link to this comment")You can easily update the example to add momentum.        - 
shahrul December 8, 2020 at 7:29 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-581816 "Direct link to this comment")can you show me how to add momentum in this tutorial, I quite confuse how to apply the momentum calculation.            - 
[Jason Brownlee](https://machinelearningmastery.com) December 9, 2020 at 6:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-581968 "Direct link to this comment")This is a common question that I answer here:\
 [https://machinelearningmastery.com/faq/single-faq/can-you-change-the-code-in-the-tutorial-to-\_\_\_](https://machinelearningmastery.com/faq/single-faq/can-you-change-the-code-in-the-tutorial-to-___)201. 
Harini October 6, 2019 at 3:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-504271 "Direct link to this comment")Dear Sir,This tutorial is really helpful for a beginner like me. I couldn’t understand where the input and output nodes are mentioned in the code. How to change number of nodes for input and output layer. Kindly help me with it.Regards    - 
[Jason Brownlee](https://machinelearningmastery.com) October 6, 2019 at 8:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-504336 "Direct link to this comment")Perhaps start with this even simpler example:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)202. 
Víctor October 21, 2019 at 5:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-506555 "Direct link to this comment")Hello Jason, Thanks for this great tutorial. How can the trained model be saved with this example? I mean with pickle or joblib in a .sav file like in other scikit classifiers. Thanks.Regards    - 
[Jason Brownlee](https://machinelearningmastery.com) October 21, 2019 at 6:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-506585 "Direct link to this comment")I recommend using sklearn for real projects, this code is for learning purposes only.That being said, you can save the “network” prepared in the backpropagation function.203. 
chamodi October 31, 2019 at 5:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-508070 "Direct link to this comment")Thank you very much sir..your articles are always very clear,greatly supporting in coding.    - 
[Jason Brownlee](https://machinelearningmastery.com) November 1, 2019 at 5:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-508169 "Direct link to this comment")Thanks!204. 
Jaya November 9, 2019 at 3:10 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-509375 "Direct link to this comment")Thanks for great tutorial. How can we determine the number of neuron we used in each of hidden layer?. Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) November 10, 2019 at 8:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-509540 "Direct link to this comment")This is a common question that I answer here:\
 [https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network](https://machinelearningmastery.com/faq/single-faq/how-many-layers-and-nodes-do-i-need-in-my-neural-network)        - 
Jaya November 13, 2019 at 12:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-510282 "Direct link to this comment")Thanks for the answer, i mean in your code is there variable that we can set to determine the sum of neuron in each hidden layer.\
 And one question more, is there case that epoch looping will stop when the error values is small or epoch looping is always looping until finish to max epoch?\
 sorry for my bad english and im a newbie in neural network\
 thanks you so much            - 
[Jason Brownlee](https://machinelearningmastery.com) November 13, 2019 at 5:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-510366 "Direct link to this comment")No, in this example we run for a fixed number of epochs.205. 
Jean November 27, 2019 at 2:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-512525 "Direct link to this comment")Hello Jason,\
 Thanks for great content.\
 Nevertheless, I think it would much better if you could also write down the mathematical equation behind the code(s). It would be much easier to understand how all “those scary math” are implemented.\
 Anyway very good job!Regards,\
 Jean    - 
[Jason Brownlee](https://machinelearningmastery.com) November 27, 2019 at 6:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-512567 "Direct link to this comment")Thanks for the suggestion.206. 
Tobias December 7, 2019 at 7:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514018 "Direct link to this comment")It does not work for xor but it works for the first data you used. Why?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 8, 2019 at 6:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514135 "Direct link to this comment")The network was designed for a specific dataset.207. 
Tobias December 7, 2019 at 7:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514024 "Direct link to this comment")this is what it is outputtingExpected\=1, Got\=0\
 Expected\=1, Got\=0\
 Expected\=0, Got\=0\
 Expected\=0, Got\=0    - 
[Jason Brownlee](https://machinelearningmastery.com) December 8, 2019 at 6:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514138 "Direct link to this comment")Perhaps try tuning the architecture and training of the model to your specific dataset.208. 
Samara Silva Santos December 8, 2019 at 1:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514104 "Direct link to this comment")Hii, I would like to know what do you mean when you say that “Using the Zero Rule algorithm that predicts the most common class value, the baseline accuracy for the problem is 28.095%.”If I use this algoritm for another use case, the accuracity is just 28%?Please, look what I have:\
 I need to modify this approach to use Quasi-newton method to calculate the error, instead of gradient method. The gradient method, what you have used, use partial derivative to calculate if the error is growing on. I see that you implemented derivative this way:def transfer\_derivative(output):\
 return output \* (1.0 – output)And what I know is that derivate is calculated this way:( f(x \+ h) – f( x) ) /hthis both way are equivalent?I already have quasi-newton method implemented but it is now really difficult to me make this modification. Please, let me know if you could help me. I really appreciate your help.    - 
[Jason Brownlee](https://machinelearningmastery.com) December 8, 2019 at 6:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514155 "Direct link to this comment")I mean predicting the majority class. It is a naive classifier sometimes called the zero rule.Sorry, I don’t have the capacity to help you adapt the example to use a different optimization algorithm.209. 
Jeff Myzek December 9, 2019 at 9:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514295 "Direct link to this comment")Hey Jason,\
 I am trying to use your code to run back propagation on MNIST with the following parameters but i am having trouble: 784 input units, a hidden layer of 100, and a Softmax group of 10 units as the output layer, cross-entropy loss objective function. I want to compute the weight update based on the entire training set, using the error backpropagation algorithm. learning rate that’s small enough for all practical purposes, but not so small that the network doesn’t learn. And I want to stop when the weight update becomes zero. Optimally i would want to see the weight vector and loss at each step. would you be able to assist me?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 9, 2019 at 1:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514314 "Direct link to this comment")I would recommend using mini-batches to approximate the error gradient.210. 
Sabarish December 11, 2019 at 6:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514593 "Direct link to this comment")Back Propagate Error:We are using the sigmoid transfer function, the derivative of which can be calculated as follows:derivative \= output \* (1.0 – output)What does it mean? I am not clear. Could you please help me understand?\
 Sigmoid function \=1/1\+e\*\*-x\
 How come derivative of it be output \* (1.0 – output)??    - 
[Jason Brownlee](https://machinelearningmastery.com) December 12, 2019 at 6:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-514652 "Direct link to this comment")The gradient or slope at a point on the function.\
 [https://en.wikipedia.org/wiki/Logistic\_function](https://en.wikipedia.org/wiki/Logistic_function)        - 
Job December 10, 2021 at 3:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-643775 "Direct link to this comment")Yes but the derivative of 1/1\+e\*\*-x\
 is equel to (e\*\*-x)/((e\*\*-x)\*\*2)\
 and not x\*(1-x)\
 is it so that the error rises as you get further from x \= 0??            - 
Adrian Tam December 10, 2021 at 4:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-643793 "Direct link to this comment")It is y \= 1/(1\+e\*\*-x)\
 and then differentiation is y’ \= y\*(1-y)211. 
Sylvan December 17, 2019 at 7:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-515516 "Direct link to this comment")Hello here!I am very new with Python in Data Science and Artificial Intelligence. Can anyone here help me with this AI assignment below due by December 19 2019, please? I am seriously stuck. Here is the question:\<> End of the question.\
 ———————————————————————Below is the indicator simple code:#Indicators.py# Import Built-Ins\
 import logging# Import Third-Party\
 import pandas as pd\
 import numpy as np# Import Homebrew# Init Logging Facilities\
 log \= logging.getLogger(\_\_name\_\_)#\
 from alpha\_vantage.timeseries import TimeSeries\
 import matplotlib.pyplot as plt# Add get\_price() def from get\_price\_alphavantagepy code\
 def get\_prices():\
 apikey \= “BW4V00IXHSAE829D” ts \= TimeSeries(key\=apikey, output\_format\=’pandas’)\
 data, meta\_data \= ts.get\_intraday(symbol\=’MSFT’,interval\=’1min’, outputsize\=’full’)\
 data\[‘4. close’\]# End add get\_price() def from get\_price\_alphavantagepy code #plt.title(‘Intraday Times Series for the MSFT stock (1 min)’)\
 #plt.show()\
 return data\[‘4. close’\] #return price#if \_\_name\_\_ \=\= “\_\_main\_\_”:\
 # get\_prices()def rsi(price, n\=14): #rsi(prices, n\=14):\
 deltas \= np.diff(prices)\
 seed \= deltas\[:n\+1\]\
 up \= seed\[seed>\=0\].sum()/n\
 down \= -seed\[seed0:\
 upval \= delta\
 downval \= 0.\
 else:\
 upval \= 0.\
 downval \= -delta\
 up \= (up\*(n-1) \+ upval)/n\
 down \= (down\*(n-1) \+ downval)/n rs \= up/down\
 rsi\[i\] \= 100. – 100./(1.\+rs)\
 return rsi\
 prices \= get\_prices()\
 print(“\n”)\
 print(rsi(prices))\
 print(“\n”)——————————–Very thank you in advance.\*Sylvan    - 
[Jason Brownlee](https://machinelearningmastery.com) December 17, 2019 at 7:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-515518 "Direct link to this comment")Perhaps try posting your code and question to stackoverflow?212. 
bismeet December 22, 2019 at 5:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516043 "Direct link to this comment")row \= \[1, 0, None\]\
 Cant understand the use of None here.    - 
[Jason Brownlee](https://machinelearningmastery.com) December 23, 2019 at 6:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516073 "Direct link to this comment")The final value in the row is the class label. Here we set None, as in no class label.        - 
bismeet December 24, 2019 at 12:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516182 "Direct link to this comment")I still don’t understand , how can an input have no class label?            - 
[Jason Brownlee](https://machinelearningmastery.com) December 24, 2019 at 4:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516196 "Direct link to this comment")In the case where we want to make a prediction.213. 
bismeet December 22, 2019 at 9:40 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516053 "Direct link to this comment")Why are there two formulas for error?error \= (expected – output) \* transfer\_derivative(output)error \= (weight\_k \* error\_j) \* transfer\_derivative(output)    - 
[Jason Brownlee](https://machinelearningmastery.com) December 23, 2019 at 6:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516077 "Direct link to this comment")They are the same, but one for the output of the model and one for credit assignment for each weight.214. 
Vaishu December 24, 2019 at 9:23 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516227 "Direct link to this comment")Why can’t we use same backpropagation algorithm code for wheatseed\_dataset as the one used in previous case?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 25, 2019 at 10:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-516270 "Direct link to this comment")You can. We do.215. 
Ansist January 19, 2020 at 1:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-518421 "Direct link to this comment")Hi, I am a little new to the implementing neural networks and the underlying mathematics. I wanted to know why the target variable (y-variable) is usually binary in nature (\[0 or 1\]). Why can’t I have, say for example returns (usually between \[-1,1\] continuous)? Secondly, is it always advised to transform your X and Y variables before feeding them into the neural network?    - 
[Jason Brownlee](https://machinelearningmastery.com) January 19, 2020 at 7:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-518447 "Direct link to this comment")You can, it is common to use 0 and 1 with a sigmoid activation function in the output layer.It is good practice to scale data:\
 [https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)216. 
[rafael gamboa](http://www.itam.mx) January 23, 2020 at 10:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-518843 "Direct link to this comment")it function perfectly!!!    - 
[Jason Brownlee](https://machinelearningmastery.com) January 23, 2020 at 12:56 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-518862 "Direct link to this comment")Thanks, I’m, happy to hear that.217. 
[Bram](https://dbrama.blogspot.com/) January 28, 2020 at 6:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-519589 "Direct link to this comment")hi jason,\
 I just finished the tutorial, this tutorial is very helpfull for me as the beginner in python and neural network. i have some question for the k-fold validationin the tutorial above I see if every fold process need to initialize a new network. Does the neural network work like that? i think the network will only be initialized once and the network will be used in the next fold? not initialize a new one. what if i use it in a real case ?\
 I might be wrong, please correct me.    - 
[Jason Brownlee](https://machinelearningmastery.com) January 29, 2020 at 6:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-519646 "Direct link to this comment")Yes, k-fold cross-validation means fitting and evaluating k different models and averaging their performance.You can learn more here:\
 [https://machinelearningmastery.com/k-fold-cross-validation/](https://machinelearningmastery.com/k-fold-cross-validation/)218. 
ssrinath February 19, 2020 at 3:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522222 "Direct link to this comment")hello jason brownleeI am a CSE student and can you help us in predicting weather using neural network backpropagation    - 
[Jason Brownlee](https://machinelearningmastery.com) February 19, 2020 at 8:07 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522272 "Direct link to this comment")Perhaps start here:\
 [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)219. 
Salvador February 20, 2020 at 9:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522419 "Direct link to this comment")Hello Jason,\
 I receive this error in spyder :\
 IndexError: list assignment index out of range.\
 Do you know where is the error?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 20, 2020 at 11:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522432 "Direct link to this comment")This will help:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)220. 
Melkamu February 22, 2020 at 1:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522771 "Direct link to this comment")Hey Jason! Thank you so much for your best tutorial    - 
[Jason Brownlee](https://machinelearningmastery.com) February 23, 2020 at 7:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522847 "Direct link to this comment")You’re welcome.221. 
Melkamu February 22, 2020 at 1:15 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522773 "Direct link to this comment")Hello Jason i am new for Python and i seriously follow your tutorial because i wanna to design my own prediction model using neural network with back propagation algorithm. but when i try to write this code on Jupiter notebook on python 3.6 “list index out of range ” error message displayed. Could you correct me ? the code i tried and error is from random import seed\
 from random import random\
 from math import expseed(1)\
 network \= initialize\_network(5, 6, 1)\
 for layer in network:\
 print(layer)\
 #Calculate neuron activation for an input\
 def activate(weights, inputs):\
 activation \= weights\[-1\]\
 for i in range(len(weights)-1):\
 activation \+\= weights\[i\] \* inputs\[i\]\
 return activation# Transfer neuron activation\
 def transfer(activation):\
 return 1.0 / (1.0 \+ exp(-activation))# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\
 for layer in network:\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 return inputs in forward\_propagate(network, row)\
 33 new\_inputs \= \[\]\
 34 for neuron in layer:\
 —> 35 activation \= activate(neuron\[‘weights’\], inputs)\
 36 neuron\[‘output’\] \= transfer(activation)\
 37 new\_inputs.append(neuron\[‘output’\]) in activate(weights, inputs)\
 20 activation \= weights\[-1\]\
 21 for i in range(len(weights)-1):\
 —> 22 activation \+\= weights\[i\] \* inputs\[i\]\
 23 return activation\
 24 IndexError: list index out of range    - 
[Jason Brownlee](https://machinelearningmastery.com) February 23, 2020 at 7:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-522849 "Direct link to this comment")Perhaps don’t use a notebook. See this:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)222. 
Pavitra February 29, 2020 at 12:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523695 "Direct link to this comment")Hello Jason,\
 I’am new to machine learning and I want to build a model using back propagation. And I am using this code for my project. This code works perfectly. But I want the prediction when I input the value so can you please tell me how can I get the prediction ?    - 
[Jason Brownlee](https://machinelearningmastery.com) February 29, 2020 at 7:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523760 "Direct link to this comment")Yes, you can change the example to just train the network then call the predict function on new data.If this is challenging for you, perhaps use a library instead, like keras:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)        - 
Pavitra February 29, 2020 at 11:12 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523802 "Direct link to this comment")Thank you so much sir.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 1, 2020 at 5:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523902 "Direct link to this comment")You’re welcome.        - 
Pavitra February 29, 2020 at 2:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523816 "Direct link to this comment")Could you please tell me how to use the predict function.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 1, 2020 at 5:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-523905 "Direct link to this comment")You can adapt the example in the tutorial directly. Sorry, I cannot prepare a custom code example for you.If it is challenging for you (which it sounds like it is), I recommend using a library instead:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)223. 
Lucas March 3, 2020 at 2:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524123 "Direct link to this comment")Hello Jason,i’ve noticed that (in chapter 4) you use the labels (0,1) as a bias constant for the bias weight multiplication in the activation function. Shouldn’t you theoratically set the all labels temporarily to 1, otherwise samples with label 0 will have no bias?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 3, 2020 at 6:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524154 "Direct link to this comment")No.Why do you think this?        - 
Lucas March 5, 2020 at 1:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524398 "Direct link to this comment")Oh nvm sorry, i missed that you add the bias explicitly “activation \= weights\[-1\]”. Most of the books I’ve read add a temporary “1” to the inputs, so that the dot product doesn’t exclude the bias. So i falsly asumed you wanted to set the labels temporarly to one “activation \= activate(neuron\[‘weights’\], inputs)” because the inputs include the labels (i already thought this would be a weird way to do it).Btw thanks for the excellent tutorial.I also tried to implement a multi-layer nn which uses np.arrays for efficent matrix multiplication. But somehow my weights get really small really fast. Is this a general problem with nn’s or is it probably a problem with my activation function?\
 I use reLu for the hidden layers and sigmoid for the output layer.            - 
[Jason Brownlee](https://machinelearningmastery.com) March 5, 2020 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524428 "Direct link to this comment")NN can be hard to debug, it could be a hyperparameter or it could be a bug in your implementation. Moving to a standard lib is highly recommended at some point.224. 
Heritiera fomes March 3, 2020 at 5:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524135 "Direct link to this comment")Hello Jason,I have to implement a placement problem, where I need to place some students in different classes, where every classes have some capacity. In that case how can I relate ANN with these?\
 If I want to add some constraints in the ANN, how can I add these constraint? for example when a test case (student) is going to be predict in which class it is assigned. My porblem needs to check the capacity of the class, then all the students must be assigned to a class.\
 It would be great help if I hear from you.    - 
[Jason Brownlee](https://machinelearningmastery.com) March 3, 2020 at 6:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524159 "Direct link to this comment")o, sounds like an constraint satisfaction / optimization problem. Look at operations research / dynamic programming.        - 
Heritiera fomes March 4, 2020 at 1:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524265 "Direct link to this comment")Thanks for your reply.If i want to add some constraints during prediction how should I do this?            - 
[Jason Brownlee](https://machinelearningmastery.com) March 4, 2020 at 5:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524309 "Direct link to this comment")Use the Keras API:\
 [https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/)225. 
A Kranthi Kiran March 4, 2020 at 9:06 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524371 "Direct link to this comment")can I know how to build a front end for this model using flask? or\
 is there any other best way to build a front end rather than flask?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 5, 2020 at 6:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524418 "Direct link to this comment")I don’t have an example, sorry.    - 
Eunike Kamase Elisabeth August 8, 2020 at 5:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-548194 "Direct link to this comment")hello, have you already know how to build the front end for backpropagation with flask?226. 
Prabhu Prasad Dev March 6, 2020 at 11:42 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524628 "Direct link to this comment")Is there any code or how to implement Spiking Neural Network(SNN).. I am very much interested to know about SNN bcoz it is the 3rd generation of neural network..Can u plz help me of details about SNN???    - 
[Jason Brownlee](https://machinelearningmastery.com) March 7, 2020 at 7:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524672 "Direct link to this comment")I don’t have an example, sorry.227. 
Namitha Dsouza March 8, 2020 at 4:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524866 "Direct link to this comment")I am new to this field. I am sorry if you do not get my question. # Train a network for a fixed number of epochs\
 def train\_network(network, train, l\_rate, n\_epoch, n\_outputs):\
 for epoch in range(n\_epoch):\
 for row in train:\
 outputs \= forward\_propagate(network, row)\
 expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1\
 backward\_propagate\_error(network, expected)\
 update\_weights(network, row, l\_rate)What is the use of these two lines? Is it only for binary classification or any classification with 3 or more classes can use this? Because this works perfectly for binary classification. But for other classifications, it gives an error. expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1    - 
[Jason Brownlee](https://machinelearningmastery.com) March 9, 2020 at 7:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-524917 "Direct link to this comment")We are one hot encoding the target class.228. 
Carlos Meza March 13, 2020 at 1:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525468 "Direct link to this comment")Hello!! Im new on this. If I want to add 9 input variables instead of 7. What do I need to change in the code in order to make it work. Amazing publication!    - 
[Jason Brownlee](https://machinelearningmastery.com) March 13, 2020 at 1:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525473 "Direct link to this comment")Perhaps start with this much simler tutorial:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)        - 
Carlos Meza March 13, 2020 at 2:14 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525476 "Direct link to this comment")Hello, thanks for your reply. I looked into it, you use \[:,0:8\] to define de input variables. However in this code is different, thats why I’m confused. Any other clue?            - 
[Jason Brownlee](https://machinelearningmastery.com) March 14, 2020 at 8:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525535 "Direct link to this comment")Ah, if you are new to list/array slicing, this will help:\
 [https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/](https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/)229. 
Alex Ramirez March 16, 2020 at 11:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525716 "Direct link to this comment")Hello! How to calculate the recall/precision/F1Score from this excersise?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 16, 2020 at 1:31 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-525724 "Direct link to this comment")You could use the sklearn library to calculate the required metrics:\
 [https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics)I don’t have the capacity to implement this for you. If it is too advanced, I strongly recommend using Keras instead:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)230. 
Sumanta Das March 30, 2020 at 3:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527523 "Direct link to this comment")How to modify the code to work with GPUs, without using fancy libraries?    - 
[Jason Brownlee](https://machinelearningmastery.com) March 30, 2020 at 5:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527551 "Direct link to this comment")Not sure you can. Fancy libraries (keras on tensorflow) let you use the GPU.231. 
Abhishek March 30, 2020 at 10:13 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527652 "Direct link to this comment")Hi Jason,Trying to execute, but I’m facing this Error. I’m running the Code on Spyder (Python 3.7)Traceback (most recent call last): File “”, line 1, in\
 runfile(‘C:/Users/duppa/Desktop/Wheat Seed Code New.py’, wdir\=’C:/Users/duppa/Desktop’) File “C:\Users\duppa\Anaconda3\lib\site-packages\spyder\_kernels\customize\spydercustomize.py”, line 786, in runfile\
 execfile(filename, namespace) File “C:\Users\duppa\Anaconda3\lib\site-packages\spyder\_kernels\customize\spydercustomize.py”, line 110, in execfile\
 exec(compile(f.read(), filename, ‘exec’), namespace) File “C:/Users/duppa/Desktop/Wheat Seed Code New.py”, line 204, in\
 scores \= evaluate\_algorithm(dataset, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden) File “C:/Users/duppa/Desktop/Wheat Seed Code New.py”, line 82, in evaluate\_algorithm\
 train\_set \= sum(train\_set, \[\]) File “C:\Users\duppa\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py”, line 2076, in sum\
 initial\=initial) File “C:\Users\duppa\Anaconda3\lib\site-packages\numpy\core\fromnumeric.py”, line 86, in \_wrapreduction\
 return ufunc.reduce(obj, axis, dtype, out, \*\*passkwargs)TypeError: ‘list’ object cannot be interpreted as an integer    - 
Abhishek March 31, 2020 at 6:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527700 "Direct link to this comment")Earlier, I tried running the code on Spyder (Python 3.7) and faced the Error of (TypeError: ‘list’ object cannot be interpreted as an integer). But when I executed the same code on Jupyter notebook I haven’t faced any Error. Output was successful        - 
[Jason Brownlee](https://machinelearningmastery.com) March 31, 2020 at 8:19 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527759 "Direct link to this comment")Happy to hear that.I recommend not using an IDE or notebook in general:\
 [https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks](https://machinelearningmastery.com/faq/single-faq/why-dont-use-or-recommend-notebooks)    - 
[Jason Brownlee](https://machinelearningmastery.com) March 31, 2020 at 8:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-527740 "Direct link to this comment")I’m sorry to hear that, this will help:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)232. 
Robin April 8, 2020 at 1:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-528787 "Direct link to this comment")Hi Jason,Just wanted to extend my thanks for your tutorial. Two months ago I wanted to learn Python and was in the middle of learning more about AI and ML and used your tutorial to help me implement my first neural network. A lot of literature and texts use a more mathematical and lower level approach to neural networks using matrices, etc which isn’t intuitive to me but your tutorial just clicked as it was easy to conceptualize and was a more simple approach.One of the ways I cement my own knowledge is writing about things I’m working on or worked on, and to really commit neural networks to memory, I wrote a tutorial myself for a neural network using learning rate and momentum parameters.[https://github.com/stratzilla/neural-network-tutorial/blob/master/neural-network-tutorial.ipynb](https://github.com/stratzilla/neural-network-tutorial/blob/master/neural-network-tutorial.ipynb)I would love to put your web page in an acknowledgements section if you were okay with that as I don’t think I would have figured out neural networks if it weren’t for your site.Regards,\
 Robin    - 
[Jason Brownlee](https://machinelearningmastery.com) April 8, 2020 at 7:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-528827 "Direct link to this comment")Thanks Robin, well done on your progress!Yes, please link back.233. 
tom April 12, 2020 at 2:39 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529462 "Direct link to this comment")Hey Jason, amazing tutorial on how to implement a 3 layer neural network with 200 lines of code. I have one question though, for the back propagation part, why is that error of (layer j, neutron i) \= summation of (weight\_k \* error\_j) \* transfer\_derivative(output)? Can you explain a little bit on mathematics? I know the error is derivatives of cost function, but how do you know the connection between error of current layer and error of next layer?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2020 at 6:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529522 "Direct link to this comment")Thanks.Sorry, I don’t dive into the theory, I recommend a good textbook like the 2016 “deep learning” or 1999 “neuralsmithing”.234. 
Quang Huy Chu April 12, 2020 at 11:51 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529494 "Direct link to this comment")Hi, first of all, thank you for posting this, it helps me very much in my Master’s research. But at my research, 4 output is required and at try to put my dataset, which is 48 samples of 16 inputs node and 4 output.\
 My question is:\
 As my research, with my small size dataset, choosing k \= dataset size (48) is needed, apply k \= 48, l\_rate \= 0.3, n\_hidden \= 42. But according to the result, the prediction is always repeated \[0,0,0,0,0,0\] ; \[2,2,2,2,2,2\] ; \[0,0,0,0,0,0\] ; \[3,3,3,3,3,3\], with different k also give the same prediction result (0,2,0,3). Can you figure it out why my NN give that strange result ?Thank you very much.    - 
Quang Huy Chu April 12, 2020 at 11:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529495 "Direct link to this comment")For example, here is my result:\
 \NN\_BP\_1.py\
 Predicted: \[0, 0, 0, 0, 0, 0\]\
 Actual: \[0, 1, 0, 3, 0, 2\]\
 Predicted: \[2, 2, 2, 2, 2, 2\]\
 Actual: \[2, 1, 2, 3, 0, 1\]\
 Predicted: \[0, 0, 0, 0, 0, 0\]\
 Actual: \[0, 2, 1, 0, 1, 3\]\
 Predicted: \[3, 3, 3, 3, 3, 3\]\
 Actual: \[1, 1, 3, 2, 0, 3\]\
 Predicted: \[0, 0, 0, 0, 0, 0\]\
 Actual: \[0, 0, 0, 1, 1, 0\]\
 Predicted: \[2, 2, 2, 2, 2, 2\]\
 Actual: \[2, 3, 1, 2, 0, 2\]\
 Predicted: \[2, 2, 2, 2, 2, 2\]\
 Actual: \[3, 2, 2, 1, 3, 3\]\
 Predicted: \[1, 1, 1, 1, 1, 1\]\
 Actual: \[3, 3, 1, 2, 3, 2\]\
 scores: \[50.0, 33.33333333333333, 33.33333333333333, 33.33333333333333, 66.66666666666666, 50.0, 33.33333333333333, 16.666666666666664\]\
 Mean Accuracy: 39.583%Can you figure it out why I have this problem or the problem is my dataset is not good?        - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2020 at 6:18 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529532 "Direct link to this comment")Perhaps try alternate model configurations?\
 Perhaps try alternate training configurations?\
 Perhaps try scaling the data?\
 Perhaps try monitoring loss during training.See these tutorials for debugging neural nets:\
 [https://machinelearningmastery.com/start-here/#better](https://machinelearningmastery.com/start-here/#better)    - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2020 at 6:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529531 "Direct link to this comment")You’re welcome.Perhaps use the Keras API instead, it will be much easier for you:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)        - 
Quang Huy Chu April 13, 2020 at 11:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529554 "Direct link to this comment")Hi Jason, thanks for the reply, after using your model and test with other datasets I found on the internet, your model works properly. Maybe it is my dataset is the problem.            - 
[Jason Brownlee](https://machinelearningmastery.com) April 13, 2020 at 1:50 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529566 "Direct link to this comment")Thanks.235. 
Bia April 14, 2020 at 5:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529635 "Direct link to this comment")Hi Jason!First, Thank you for this good tutorial, it really helps a lot.\
 My question is that in your code you just use one hidden layer but how to add more hidden layers in same above code. Kindly guide me as I am a beginner. Thank you:)    - 
[Jason Brownlee](https://machinelearningmastery.com) April 14, 2020 at 6:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529672 "Direct link to this comment")You’re welcome.If you are a beginner, I recommend starting here:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)236. 
audrey April 14, 2020 at 5:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529640 "Direct link to this comment")did you find something about Relu? i need this too thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) April 14, 2020 at 6:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529675 "Direct link to this comment")See thus tutorial:\
 [https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)237. 
Alex April 16, 2020 at 4:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529896 "Direct link to this comment")If the transfer derivative is \= output \* (1.0 – output), when calculating errors in a hidden layer, which has a bias \= 1, doesn’t that mean the transfer derivative is always 1 \* (1-1) \= 0 for the bias node? Therefore the error of the bias node is always 0 because error \= (weight\_k \* error\_j) \* transfer\_derivative(output)?If that’s true, then the weights from the bias node never update because you multiply by the error.I assume I’m missing something. How do the weights from a bias node get updated (i.e. how is the error of the bias ever not 0)?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 16, 2020 at 6:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529924 "Direct link to this comment")The implementation is based on the description in this book if you’d like to know more:\
 [https://amzn.to/3en9SPL](https://amzn.to/3en9SPL)238. 
Alex April 16, 2020 at 7:18 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529927 "Direct link to this comment")I guess I’m not so much interested in why at the moment, but that in your implementation, which as far as I can tell is correct (it matches logically identically with a C/C\+\+ I came up with), there is no way to update the weights coming off the bias.This line \[ neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\]) \] seems as if it will always result in neuron\[‘delta’\] \=\= 0 for the bias. I noticed it my C\+\+ implementation and when I went looking for answers, came across your post, and it looks like yours would result in 0 also.So I’m more interested in if you found that to be the case. If so, it can escape detection because the network will still learn, just not as well or as fast, so with toy data this will not be noticed.    - 
[Jason Brownlee](https://machinelearningmastery.com) April 16, 2020 at 1:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-529954 "Direct link to this comment")Interesting, thanks for sharing.I have not observed this issue, have you confirmed that indeed bias weights in the above implementation are unchanged after initialization?        - 
Alex June 4, 2020 at 3:47 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537886 "Direct link to this comment")Sorry for the long delay in answering, I got distracted, but remembered to come back to this. I did look at the weights and you are correct, they do update. However, there is still something missing (admittedly, most likely in my understanding). You create a bias node by adding 1 weight vector per layer (weights from bias to next level neurons), but I do not see anywhere where the bias node activation it is explicitly set to 1. It seems that the activation of the bias node is treated like all other nodes, and is free to change value (I printed them during training and they are never 1.0). So the weights are also updating because bias does not equal 1.If the bias activation is forced to stay at 1 (which seems correct for the algorithm), the weights from the bias cannot update because in the transfer derivative: 1 \* (1-1) \= 0. I did try that as well, and it shows the weights do not update. If I’m correct, this seems like a very subtle flaw, which would be undetectable in any simple learning problem because the network will learn and predict with or without a bias (I tried that too, and it does work either way).With that said, I’m still not convinced I’m correct. I might still be missing something, but after several hours in the code, I can’t find any way the bias activation holds to 1.0. When it isn’t 1.0, the bias weights will update because the transfer derivative is non-zero, but that violates the role of the bias node. When the bias activation is forced to 1.0 which is the correct value for the bias, the weights do not update because 1 \* (1-1) \= 0. So I’m still confused, but still open to the possibility I’m just not understanding something about the code.            - 
[Jason Brownlee](https://machinelearningmastery.com) June 4, 2020 at 6:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537930 "Direct link to this comment")To understand the bias consider the forward and backward pass.For the forward pass, see the activate() function and notice that the bias activation (stored as the last element of the list) is added to the activation first. The same as 1\*bias\_weight.For the backward pass, the update\_weights() function update the bias weight first, then the other weights.Perhaps re-read the text and code of the tutorial. This is all discussed.239. 
Maria Campero April 17, 2020 at 2:15 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-530043 "Direct link to this comment")Hi Jason,\
 First of all thank you very much for this post it is really helpful,\
 At the moment I’m writing a code for neural network with backpropagation in Phyton. I have 8 inputs and 7 outputs with one hidden layer(1neuron). I scaled the dataset then tryed to used your code but I have error of alignment. Can you please give me some advice to fix that issue    - 
[Jason Brownlee](https://machinelearningmastery.com) April 17, 2020 at 6:24 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-530082 "Direct link to this comment")This is an advanced tutorial and it sounds like you are having trouble. I recommend using the Keras API instead, it’s much simpler:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)240. 
sameer sakkhari April 25, 2020 at 1:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531547 "Direct link to this comment")I have a dataset isolet in the form of csv file. I want to implement a Neural Network with backpropagation in python using tensorflow . How do I start ? How do I load my data?You said to save the dataset in csv format in current working directory. But it is not able to recognize isolet    - 
[Jason Brownlee](https://machinelearningmastery.com) April 25, 2020 at 7:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531591 "Direct link to this comment")This will show you how to load your data:\
 [https://machinelearningmastery.com/load-machine-learning-data-python/](https://machinelearningmastery.com/load-machine-learning-data-python/)241. 
João Guilherme Cotta April 26, 2020 at 2:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531694 "Direct link to this comment")Hello Jason,Thanks for this tutorial, it is very helpful.I would like to modify this code to use MLP with BP to predict the velocity of a car based on different inputs, such as velocity, acceleration, pedal position, etc. I am having some difficulty adapting the ‘expected’ part of the code, since in your example you are using only zeroes and ones, and my study case would have different values of velocity given by the dataset.Do you have any advice regarding this?Thanks in advance.    - 
[Jason Brownlee](https://machinelearningmastery.com) April 26, 2020 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531728 "Direct link to this comment")Yes, as a beginner, I strongly recommend using Keras instead:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)242. 
Sid April 26, 2020 at 1:30 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531778 "Direct link to this comment")Hi, I used the updated dataset provided and copied the code exactly(copy/paste) to test. However, when I run the code I get the error “TypeError: ‘list’ object cannot be interpreted as an integer”. Do you know why this may be happening?    - 
[Jason Brownlee](https://machinelearningmastery.com) April 27, 2020 at 5:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-531892 "Direct link to this comment")Sorry to hear that you are having trouble, perhaps this will help:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)243. 
Q.H.Chu May 1, 2020 at 12:24 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-532741 "Direct link to this comment")Hi Jason, According to the post, Your Network is using only 1 Hidden layer (Maybe its called Shallow Feed forward NN) , is that hidden layer represent Logistic Regression step? And is there a way to add more hidden-layer, many hidden-layer will improve the accurate of netowkAnd one more question, How can I choose a fit parameter (epoch, learning rate or hidden layer neurons number) for this network ? does it depends on output and input neurons?Once again, thank you very much for posting this helpful post and looking for see your reply    - 
[Jason Brownlee](https://machinelearningmastery.com) May 1, 2020 at 2:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-532753 "Direct link to this comment")Yes, one hidden layer. No not logistic regression.Tune the parameters of your model to your data.244. 
Ahmed Gad May 7, 2020 at 9:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-533620 "Direct link to this comment")Hi,Please help with adapting the code to include Upper and Lower weights for each neuron without biases \=\=> Rough Neural Network.Thanks!    - 
[Jason Brownlee](https://machinelearningmastery.com) May 7, 2020 at 11:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-533638 "Direct link to this comment")This is a common question that I answer here:\
 [https://machinelearningmastery.com/faq/single-faq/can-you-change-the-code-in-the-tutorial-to-\_\_\_](https://machinelearningmastery.com/faq/single-faq/can-you-change-the-code-in-the-tutorial-to-___)245. 
Ahmed Gad May 7, 2020 at 9:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-533623 "Direct link to this comment")The RNN structure replaces the traditional neuron by two neurons (lower neuron, upper neuron ) to represent lower and upper approximations of each attribute in the CTG data set, Its structure formed from 4 layers input, 2 hidden and output layers. The hidden layers have rough neurons which overlap and exchange information between each other, While the input and output layers consists of traditional neurons as in the figure(1):This image illustrates the idea more: [https://imgur.com/AZ0FTbY](https://imgur.com/AZ0FTbY).    - 
[Jason Brownlee](https://machinelearningmastery.com) May 7, 2020 at 11:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-533639 "Direct link to this comment")Thanks for sharing.        - 
Ahmed Gad May 8, 2020 at 10:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-533836 "Direct link to this comment")Please i need help where and how to customize the code!246. 
John Pillar May 10, 2020 at 4:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-534149 "Direct link to this comment")HI Jason – thanks very much for a wonderfully clear and understandable description. I really appreciate your ‘gentle’ approach. I’ve re-coded your python into C – I find it’s the best way for me to really learn what’s going on. Please – I have a couple of question – to apply softmax to the output – it’s easy enough to map the outputs using softmax so that they are ‘probabilities’ that sum to one, but – what changes do I need to make to the transfer function derivative in the backpropagation code. I’ve read several descriptions that say that backpropagation of the output layer errors after softmax follows exactly the same as sigmoid – so I’m confused. I think it should be different, but I may be missing something.Also – cross-entropy loss is commonly described as a natural ‘partner’ to softmax, but actually, in practice, is the ‘error’ still (expected\_value) – (predicted value) , just like you have in your code?Thanks very much if you have time to consider my question – much appreciated.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 11, 2020 at 5:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-534201 "Direct link to this comment")You’re welcome.Yes, softmax error would be the same, except calculated for each output node.Yes, you can see an implementation of cross entropy here:\
 [https://machinelearningmastery.com/cross-entropy-for-machine-learning/](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)        - 
John May 18, 2020 at 6:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-535186 "Direct link to this comment")Thanks Jason – appreciate your help.            - 
[Jason Brownlee](https://machinelearningmastery.com) May 18, 2020 at 6:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-535187 "Direct link to this comment")You’re welcome.247. 
Sandeep Kumar Dash May 29, 2020 at 9:41 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537103 "Direct link to this comment")Hi. Thanks for the great tutorial. How can I save the network diagram in file?    - 
[Jason Brownlee](https://machinelearningmastery.com) May 30, 2020 at 5:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537159 "Direct link to this comment")THanks.Perhaps use Keras and see this tutorial:\
 [https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/](https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/)248. 
Andirian Ahmad May 30, 2020 at 5:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537220 "Direct link to this comment")Hello sir, sorry for disturbance, may i ask, can we use other datasets instead of seeds dataset for this BPNN algorithm?    - 
[Jason Brownlee](https://machinelearningmastery.com) May 31, 2020 at 6:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-537325 "Direct link to this comment")Sure!249. 
Anon June 9, 2020 at 6:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-538800 "Direct link to this comment")Hi Jason. Thank you so much for this tutorial. I would just like to know if this would work with the iris dataset?b    - 
[Jason Brownlee](https://machinelearningmastery.com) June 10, 2020 at 6:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-538878 "Direct link to this comment")Sure.250. 
Zach June 11, 2020 at 5:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-539031 "Direct link to this comment")Do you perhaps have a Java or C# version of this code? I’m trying to understand it in OOP principles and have done up to the end of prediction, but the last portion just confuses me    - 
[Jason Brownlee](https://machinelearningmastery.com) June 11, 2020 at 6:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-539065 "Direct link to this comment")Sorry I do not.251. 
Muhammad Basit Umair June 11, 2020 at 11:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-539136 "Direct link to this comment")Sir kindly guide me about difference between “multilayer feed-forward network” and deep neural network (DNN).\
 Or can we say that, a multilayer feed-forward network is a deep neural network?Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) June 12, 2020 at 6:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-539173 "Direct link to this comment")MLP can be made deep by adding many layers, so can a CNN, LSTM or any type of network.252. 
Nasimul June 28, 2020 at 8:05 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-541717 "Direct link to this comment")I am getting this error, please help File “F:/khaise/neuralnet.py”, line 80, in activate\
 activation \+\= weights\[i\] \* inputs\[i\]TypeError: can’t multiply sequence by non-int of type ‘float’    - 
[Jason Brownlee](https://machinelearningmastery.com) June 29, 2020 at 6:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-541804 "Direct link to this comment")I’m sorry to hear that, this may help:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)253. 
navid July 7, 2020 at 2:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-543043 "Direct link to this comment")Hello , i have question. this code only support 1 hidden layer?    - 
[Jason Brownlee](https://machinelearningmastery.com) July 7, 2020 at 6:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-543078 "Direct link to this comment")You can extend it to add more layers.254. 
DavidHE July 24, 2020 at 1:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-545766 "Direct link to this comment")I’m trying to implement this tutorial in a language different than python, If during the training of the net the value of the varibale sum\_error gets stuck or even goes up a little and down again, that means an error in the implementation?    - 
[Jason Brownlee](https://machinelearningmastery.com) July 24, 2020 at 1:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-545771 "Direct link to this comment")Perhaps run the same code with the same initial weights and compare the output of each step?255. 
Shyam August 6, 2020 at 3:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-547887 "Direct link to this comment")Hi, How can I get a loss vs epoch graph for this code?    - 
[Jason Brownlee](https://machinelearningmastery.com) August 7, 2020 at 6:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-547952 "Direct link to this comment")Yes, you will have to implement it yourself though and use a train/test split instead of k-fold cross-validation.256. 
[arun](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) August 8, 2020 at 4:01 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-548179 "Direct link to this comment")Hai In Initialize Network function hidden\_layer variable store three random weight ,but we are given one hidden layer is used one weight for feed forward another one weight backward remaining one which propose used ??. similarly output\_layer got three weight i did not under can you explain    - 
[Jason Brownlee](https://machinelearningmastery.com) August 9, 2020 at 5:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-548278 "Direct link to this comment")Sorry, I don’t follow. Can you please rephrase or elaborate on your question?257. 
[arun](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) August 8, 2020 at 8:09 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-548210 "Direct link to this comment")Hi i need predicted \= algorithm(train\_set, test\_set, \*args) behind this line operation ?    - 
[Jason Brownlee](https://machinelearningmastery.com) August 9, 2020 at 5:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-548287 "Direct link to this comment")Sorry, what do you mean exactly?258. 
Nate August 14, 2020 at 9:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-549125 "Direct link to this comment")do you know how to extract the “deltas” for each input and synaptic weight?    - 
[Jason Brownlee](https://machinelearningmastery.com) August 14, 2020 at 1:18 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-549150 "Direct link to this comment")Yes, the element added to each weight would be the deltas as you describe them.        - 
Nate August 15, 2020 at 12:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-549275 "Direct link to this comment")But how would you extract them in the code? I want to print the weight before and after the delta was added. What part of the code would you modify?            - 
[Jason Brownlee](https://machinelearningmastery.com) August 15, 2020 at 6:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-549336 "Direct link to this comment")You could retrieve them from the part of the code that updates the model weights, in the backward\_propagate\_error function I guess.259. 
Niloo September 5, 2020 at 9:22 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-559749 "Direct link to this comment")Hi\
 I am interested in machine learning. I read your code at this website and now I am willing to add some features so I have a question. How can we add more layer to this neural network or in the other words how can we make the number of hidden layers flexible? Could you please explain me or send me a link to learn more?\
 Thanks for your attention.    - 
[Jason Brownlee](https://machinelearningmastery.com) September 6, 2020 at 6:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-559898 "Direct link to this comment")Perhaps it would be easier for you to start with the Keras API here:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)260. 
Harsha September 25, 2020 at 8:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-564682 "Direct link to this comment")Oh Legend!!You are extremely awesome – I can’t thank you enough.As an aspiring ML engineer this is what all I needed. You will be remembered forever as the mentor who taught me ANN from scratch    - 
[Jason Brownlee](https://machinelearningmastery.com) September 25, 2020 at 9:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-564684 "Direct link to this comment")Thanks.261. 
Jon October 15, 2020 at 3:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-567425 "Direct link to this comment")I tried to convert this example to use ReLU by changing the transfer function to be:def transfer(activation):\
 return 0.0 if activation \<\= 0.0 else activationand the transfer\_derivative to be:def transfer\_derivative(output):\
 return 0.0 if output \<\= 0.0 else 1.0This seem to break the training system however and the error is never reduced.Any thoughts?Thanks for a great article anyway.    - 
[Jason Brownlee](https://machinelearningmastery.com) October 15, 2020 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-567466 "Direct link to this comment")Perhaps try cross entropy loss.\
 Perhaps try changing the model architecture.\
 Perhaps try changing the learning hyperparameters.        - 
Jon October 16, 2020 at 5:31 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-567928 "Direct link to this comment")Ok thanks Jason, sounds interesting, I’ll certainly take a look.262. 
Dinesh Kumar October 17, 2020 at 10:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-568104 "Direct link to this comment")Hi Jason,Thanks for your help to understand the Back-props concepts with python. could you please help me how we will implement based computational graphex: [https://i.imgur.com/0xUaxy6.png](https://i.imgur.com/0xUaxy6.png)    - 
[Jason Brownlee](https://machinelearningmastery.com) October 17, 2020 at 1:43 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-568157 "Direct link to this comment")I recommend using the Keras framework:\
 [https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)263. 
[JG](https://acehl.org/) October 31, 2020 at 10:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-572022 "Direct link to this comment")Hi Jason,I decided to follow this “old” tutorial, by the possibility of understand at low level the main AI’s functions such as network model definition (through a list of weights per neuron and layers on dictionary), how to activate manually the inputs and output of a neuron), network forward input propagation, but specially for me the core of the AI nets: the back error propagation), etc.Finally I decided to jump into high level Api Keras model, to wrapper of these detailed functions into a more integrated ones such as Model/Sequential, with their methods of .fit, .evaluate, .predict, and tools such as to\_categorical, etc.Besides sklearn libraries for normalization, kfold, onehotencoding, etc. Of course I got better accuracies (97.5% as mean kfold) because I could used “relu”, activations functions, and output her layers types, etc… So one more time, thanks for this tutorial to have the chance to understand better the motor that it is running below tensorflow and specially under Keras High level structures…    - 
[Jason Brownlee](https://machinelearningmastery.com) November 1, 2020 at 7:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-572085 "Direct link to this comment")Nice work!The tutorial really should be updated to use cross entropy and relu, e.g. modern ideas. I wrote this implementation like we used to in the 90s.264. 
[JG](https://acehl.org/) October 31, 2020 at 10:52 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-572025 "Direct link to this comment")More particularly the “backpropagation” ML concept, I rather prefer to cal it more intuitively and personal as “distribution of output error between all the weights / biases of neurons of all layers of the model” . So “error’s distribution” between all errors contributors (model’s weights/biases) it is for me a better name and key idea than standard one of “backpropagation”…    - 
[Jason Brownlee](https://machinelearningmastery.com) November 1, 2020 at 7:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-572086 "Direct link to this comment")Agreed. That is the key learning from this tutorial!265. 
Lia Jusmai Theresia November 7, 2020 at 5:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573471 "Direct link to this comment")Hello, can you help me create a rainfall prediction code using the neural network in python?\
 I do not understand. I have monthly data spanning 10 years. The total data that I got was 120 data. How about the input, hidden, output layer? How do you get neron? and what parameters will be used? Thank you in advance    - 
[Jason Brownlee](https://machinelearningmastery.com) November 8, 2020 at 6:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573582 "Direct link to this comment")I recommend starting with the tutorials here:\
 [https://machinelearningmastery.com/start-here/#deep\_learning\_time\_series](https://machinelearningmastery.com/start-here/#deep_learning_time_series)266. 
[Arnav](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) November 7, 2020 at 6:32 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573495 "Direct link to this comment")Can you fix my code please?\
 [https://colab.research.google.com/drive/1Skfq3A1u7Mwdo72YBRWOm4x0SCp8mIFn?usp\=sharing](https://colab.research.google.com/drive/1Skfq3A1u7Mwdo72YBRWOm4x0SCp8mIFn?usp=sharing)    - 
[Jason Brownlee](https://machinelearningmastery.com) November 8, 2020 at 6:39 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573584 "Direct link to this comment")This is a common question that I answer here:\
 [https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code](https://machinelearningmastery.com/faq/single-faq/can-you-read-review-or-debug-my-code)267. 
Pedro H November 9, 2020 at 5:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573856 "Direct link to this comment")Hi, is this code rigged for the 2:1:2 layout?If yes can you point me some good articles to better understand the back forward prop?Anyway, great work! It REALY helped me!    - 
[Jason Brownlee](https://machinelearningmastery.com) November 9, 2020 at 6:16 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573871 "Direct link to this comment")Not really.You can adapt the architecture of the model directly.268. 
[Chris Mahoney](https://www.linkedin.com/in/chrimaho/) November 9, 2020 at 9:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573901 "Direct link to this comment")Hi Jason,I LOVED this article. It helped me immensely in learning about the intricacies of Neural Networks and Deep Learning in recently months. Thank you so much!I note here that you do a node-by-node method of implementation. But there is also another method using matrix multiplication and linear algebra.I’ve taken these concepts and processes, and written up a similar article. Except, I’ve used R and the matrix method. I’d love to know your thoughts on it:\
 [https://towardsdatascience.com/vanilla-neural-networks-in-r-43b028f415?sk\=f47b3d6f9f539e907d272966fa88bcb8](https://towardsdatascience.com/vanilla-neural-networks-in-r-43b028f415?sk=f47b3d6f9f539e907d272966fa88bcb8)Thank you again for your assistance. It has helped me greatly!Cheers,\
 Chris M    - 
[Jason Brownlee](https://machinelearningmastery.com) November 9, 2020 at 1:15 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-573954 "Direct link to this comment")Thanks.Well done.269. 
Joey Hung November 22, 2020 at 3:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-577402 "Direct link to this comment")Hi Jason,Thanks for your code.\
 However, when I was running it, it has below problem and I don’t know how to fix it. Could you help to fix it?Traceback (most recent call last):\
 File “MDSHW3-2.py”, line 187, in\
 scores \= evaluate\_algorithm(df, back\_propagation, n\_folds, l\_rate, n\_epoch, n\_hidden)\
 File “MDSHW3-2.py”, line 59, in evaluate\_algorithm\
 train\_set.remove(fold)\
 ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()    - 
[Jason Brownlee](https://machinelearningmastery.com) November 22, 2020 at 6:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-577460 "Direct link to this comment")Sorry to hear that, these tips may help:\
 [https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me](https://machinelearningmastery.com/faq/single-faq/why-does-the-code-in-the-tutorial-not-work-for-me)270. 
Chiso Buso December 1, 2020 at 7:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-579861 "Direct link to this comment")Thank you very much, I would be glad if I can get python code for ANN-Using BP for a regression problem. Like the inputs of 10 parameters and outputs of the continuous value of 5 parameters. Look forward to hearing from you.    - 
[Jason Brownlee](https://machinelearningmastery.com) December 1, 2020 at 8:06 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-579863 "Direct link to this comment")The above example can be adapted for your regression problem directly.271. 
Lokman Hakim December 5, 2020 at 2:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-580871 "Direct link to this comment")Can i know why we pass ‘None’ in the code “row \= \[1,0, None\]” which is in the forward propagation phase.\
 Is it because it is related to bias?    - 
[Jason Brownlee](https://machinelearningmastery.com) December 5, 2020 at 8:09 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-580953 "Direct link to this comment")We mark the label as None.272. 
[Sina Birecik](http://sinabirecik.wordpress.com) December 21, 2020 at 9:17 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587058 "Direct link to this comment")Hello Jason,\
 At first, many thanks for the tutorial, it has cleared a lot of things about DL in my mind.\
 I would like to make a contribution. It includes the code that allows your network to get deeper. People here can upgrade your code to multi hidden layers, each neuron number in each hidden layer can be adjusted. Just follow the instructions:\
 1) Copy the code in the link below and overwrite the whole “initialize\_network” method.\
 [https://pastebin.pl/view/fc96c453](https://pastebin.pl/view/fc96c453) \
 2) Replace any “n\_hidden” term with “hidden\_list”.\
 3) At the end of the code, you can adjust hidden layers like the example below:\
 Example:\
 hidden\_list \= \[5, 3, 7\]\
 means there are 3 hidden layers, each hidden layer has 5, 3 and 7 neurons, respectively.Best regards.    - 
[Jason Brownlee](https://machinelearningmastery.com) December 21, 2020 at 1:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587147 "Direct link to this comment")You’re welcome.Thanks for sharing!273. 
Frank December 22, 2020 at 12:01 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587510 "Direct link to this comment")Hello, I have a simple question, your data have 3 classes like 1-2-3 , for example if we have different number of classes and if they are string like YES NO , do we need to convert them to 0-1 or it does not matter . I checked the code for find the answer but i could not be able to find it. Please answer me .    - 
[Jason Brownlee](https://machinelearningmastery.com) December 22, 2020 at 1:36 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587552 "Direct link to this comment")Yes, class labels should be integer encoded then one hot encoded prior to modeling for neural networks.Perhaps start here:\
 [https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)274. 
[GreakBoy](http://xn--odt-joa) December 23, 2020 at 5:28 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587808 "Direct link to this comment")Hı, ıt was a really good article , thank you for doing such a good work.\
 I have a question , in your csv data your all coloumns are integer and your classes are at the last coloum, numbers but what if\
 we have data csv like that and one class is Mercedes and the other is Porsche in this situation do we need to implement any extra code in your example code to convert, ı tried it still works but i am not sure about answer.5001700134,2053150024,961776886,88349551,15594,793434083,Mercedes\
 4363829956,1773486023,8596657562,874662638,12190,763556063,Porsche    - 
[Jason Brownlee](https://machinelearningmastery.com) December 23, 2020 at 5:37 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-587820 "Direct link to this comment")Thanks.Perhaps scale the values prior to model and integer encode labels.275. 
Gloria January 22, 2021 at 8:38 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-594015 "Direct link to this comment")Thanks for this useful article!\
 I want to change Sigmoid function into ReLU, so I modified the following 2 functions: (1) transfer(activation) and (2) transfer\_derivative(output) as follows:(1) return 1.0 / (1.0 \+ exp(-activation)) \=> return max(activation,0)\
 (2) return output \* (1.0 – output) \=> return 1 if output>0 else 0However, the network isn’t learning (accuracy \~33.3% for each fold, even when I train more epoches). Did I get something wrong? Thanks in advance!    - 
[Jason Brownlee](https://machinelearningmastery.com) January 23, 2021 at 7:03 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-594072 "Direct link to this comment")You’re welcome.Nice work!Perhaps check you didn’t change the activation function in the output layer!\
 Perhaps change the loss to cross entropy?\
 Perhaps change the architecture?\
 Perhaps change the learning hyperparameters?        - 
Gloria January 25, 2021 at 7:18 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-594350 "Direct link to this comment")Hi Jason, thanks for the reply!\
 I’ve checked the activations and weights, and found that it is the problem of ‘dying ReLU’. Some units (in this case the 3 output unit) always output 0 and cannot recover with further training.            - 
[Jason Brownlee](https://machinelearningmastery.com) January 26, 2021 at 5:50 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-594465 "Direct link to this comment")Try an alternate weight initialization, like “he”.Try scaling inputs to the range 0-1.276. 
Unnikrishnan February 11, 2021 at 1:28 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597108 "Direct link to this comment")Thanks Jason. Nice article.\
 Concepts of Backpropagation became clear now.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 12, 2021 at 5:43 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597186 "Direct link to this comment")Thank you, I’m happy it helps!277. 
AbdulAhad February 14, 2021 at 1:00 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597421 "Direct link to this comment")Thanks for the great article. Still helping who want to know exactly what and how it happens from coding presepective.    - 
[Jason Brownlee](https://machinelearningmastery.com) February 14, 2021 at 5:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597443 "Direct link to this comment")You’re welcome!278. 
David February 15, 2021 at 6:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597509 "Direct link to this comment")Hi Jason,Thanks a bunch for this. Please, can this code work in Python 3.x?Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) February 15, 2021 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-597523 "Direct link to this comment")You’re welcome David!Yes, the code works with Python 3.279. 
Lukasz March 12, 2021 at 9:25 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-600675 "Direct link to this comment")Hi Jason, Thank you for great article! I run the code on my own dataset where I was predicting three label classes. The algorithm gave me a great results with the accuracy of the prediction above 97%. Right now I am trying to use the trained network to predict the results of the new dataset, where I would not provide any labels to calculate the accuracy. Do you have any recommendation for me? Thank you!    - 
[Jason Brownlee](https://machinelearningmastery.com) March 13, 2021 at 5:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-600730 "Direct link to this comment")You can remove the evaluation of the model, fit the model on all available data and call predict on new data.280. 
[Gordon](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) March 16, 2021 at 2:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-600975 "Direct link to this comment")Hi Jason, why do we copy the row and set row\[-1\] to None?\
 This is in the function evaluate\_algorithm:for row in fold:\
 row\_copy \= list(row)\
 test\_set.append(row\_copy)\
 row\_copy\[-1\] \= Noneit seems like you could just dofor row in fold:\
 test\_set.append(row)Thanks for your help!    - 
[Jason Brownlee](https://machinelearningmastery.com) March 16, 2021 at 4:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-600999 "Direct link to this comment")So that the expected output value is not available to the model.        - 
vitor January 28, 2024 at 8:47 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-708087 "Direct link to this comment")but why on inicialization we do:n\_inputs \= len(row) – 1then?            - 
James Carmichael January 29, 2024 at 7:05 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-708153 "Direct link to this comment")Hi vitor…Please clarify the code portion you are referring to. Also, are you experiencing an error with the code provided? That will enable us to better guide you.281. 
mlhan March 27, 2021 at 2:58 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-602099 "Direct link to this comment")Do u have any idea if I want the user enters the input.how can I do it 🙁    - 
[Jason Brownlee](https://machinelearningmastery.com) March 29, 2021 at 5:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-602446 "Direct link to this comment")Yes, but this is a programming question, not a machine learning question.Perhaps you can develop a program/interface around your model.282. 
Ashwini April 16, 2021 at 5:18 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-605154 "Direct link to this comment")Hi,I tried adding softmax activation function in the output layer for doing a multi-class classification.# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\
 #hidden layer\
 layer \= network\[0\]\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 #output layer\
 layer \= network\[1\]\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= softmax(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 return inputsBut when I’m running this the accuracy of the model is falling drastically from 77% to 35%.\
 Can you please suggest me why this is happening or any other additional changes should i do to maintain the accuracy    - 
[Jason Brownlee](https://machinelearningmastery.com) April 16, 2021 at 5:34 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-605180 "Direct link to this comment")This may help you with understanding softmax:\
 [https://machinelearningmastery.com/softmax-activation-function-with-python/](https://machinelearningmastery.com/softmax-activation-function-with-python/)283. 
redouane kassa April 21, 2021 at 11:34 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-606351 "Direct link to this comment")Hello,\
 The derivative of sigmoid function should be f'(x)\=f(x)\*(1-f(x)) and not f'(x)\=x(1-x). am I right?\
 Thank you    - 
[Jason Brownlee](https://machinelearningmastery.com) April 22, 2021 at 5:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-606404 "Direct link to this comment")Yes, that is what we use. output is not x, output is f(x), e.g. f(x)\=output.284. 
James May 22, 2021 at 11:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-610371 "Direct link to this comment")Scores: \[90.47619047619048, 92.85714285714286, 97.61904761904762, 92.85714285714286, 92.85714285714286\]\
 Mean Accuracy: 93.333%    - 
[Jason Brownlee](https://machinelearningmastery.com) May 23, 2021 at 5:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-610427 "Direct link to this comment")Well done.285. 
winter May 27, 2021 at 11:53 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611446 "Direct link to this comment")Hello , Jason\
 Really thank you for your effort and nice resource.can I get some idea to visualize error, and validation like keras ?It would be a great help if I get some ideas.    - 
[Jason Brownlee](https://machinelearningmastery.com) May 28, 2021 at 6:48 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611545 "Direct link to this comment")You’re welcome.Yes, you can create learning curves if you wish, perhaps use matplotlib to create the plots.286. 
[sudip](http://laudarisudip.com.np/) May 30, 2021 at 11:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611761 "Direct link to this comment")Hello Jason,\
 Really thank you for the nice tutorial. I am using your tutorial to train my time series data where train values are almost similar for all the classes. Your tutorial gives pretty nice result. i am trying to visualize error and accuracy like keras but could’s figure out.\
 By chance is there any tutorial or sources so that I could visualize my training informatin?    - 
[Jason Brownlee](https://machinelearningmastery.com) May 31, 2021 at 5:49 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611783 "Direct link to this comment")You’re welcome.Perhaps you can plot the expected time series values as a line plot and plot the predicted values for the same time period on the same plot to provide a visual comparison of the values.287. 
[sudip](http://laudarisudip.com.np) May 31, 2021 at 5:09 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611814 "Direct link to this comment")Hi Jason,,\
 I tried to follow your tutorial to visualize error,\
 Whenever I tried to plot it say that ‘ expected is not define’\
 In my understanding ‘expected’ is defined in a function where we used backpropagate.Can I get some idea to solve this problem and plot error line?123456789101112131415161718from matplotlib import pyplotfrom sklearn . metrics import mean\_squared \_error# calculate errorserrors  \=  list ( )for  i  in  range ( len ( expected ) ) : # calculate errorerr  \=  ( expected \[ i \]  -  predicted \[ i \] ) \* \* 2 # store errorerrors . append ( err ) # report errorprint ( '>%.1f, %.1f \= %.3f'  %  ( expected \[ i \] ,  predicted \[ i \] ,  err ) )# plot errorspyplot . plot ( errors )pyplot . xticks ( ticks \= \[ i  for  i  in  range ( len ( errors ) ) \] ,  labels \= predicted )pyplot . xlabel ( 'Predicted Value' )pyplot . ylabel ( 'Mean Squared Error' )pyplot . show ( )    - 
[Jason Brownlee](https://machinelearningmastery.com) June 1, 2021 at 5:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-611855 "Direct link to this comment")You must make predictions before you can plot them and their error.288. 
sacin June 19, 2021 at 4:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-613687 "Direct link to this comment")Had a doubt,in section 3.2 shouldn’t, error \= (expected – output) \* transfer\_derivative(output)\
 be,\
 error \= (output – expected) \* transfer\_derivative(output)was thinking if this is flipped then the weights in the hidden layers might increase instead of decrease and vice versa.was referring to [https://coursera.org/share/3046ebc8c09a4bf792b4a00848f23c6c](https://coursera.org/share/3046ebc8c09a4bf792b4a00848f23c6c) by andrew NG.    - 
[Jason Brownlee](https://machinelearningmastery.com) June 19, 2021 at 5:56 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-613719 "Direct link to this comment")It can be, it just changes the direction/sign.289. 
[SUDIP LAUDARI](http://laudarisudip.com.np/) June 20, 2021 at 10:57 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-613906 "Direct link to this comment")HI Jason, still couldn’t plot smooth graph like keras does. It would be a great help if we can get an example from you.Thanks    - 
[Jason Brownlee](https://machinelearningmastery.com) June 21, 2021 at 5:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-613935 "Direct link to this comment")You must adapt the example to not use cross-validation, but a train/test split instead.Then evaluate the model’s performance on a training set and validation set each epoch (iteration).Sorry, I don’t have the capacity to prepare an example for you.290. 
[AIbird](http://none.com) August 30, 2021 at 10:57 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-624919 "Direct link to this comment")Hi Jason,\
 I am using your code for my project. Looks great. It was working perfectly before. I changed my datasets format where values look almost similar and they are in the the range of 0.05 to 0.12. Training CSV which contains all the values have around 300 rows and 120 columns.Now my error doesn’t converge to near 0 . It is always like 89 or 91.\
 I plotted all n\_folds accuracy with epoch, looks so fluctuation in accuracies. Is there any idea or suggestion to make error near to 0 so that I can expect high mean accuracy?Any idea or suggestion would be really appreciated.Thank you    - 
Adrian Tam September 1, 2021 at 8:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-625340 "Direct link to this comment")Would you try to use a scaler at preprocessing stage?291. 
[AIbird](https://www.showwcase.com/sdip) September 1, 2021 at 10:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-625389 "Direct link to this comment")Hello Adrian\
 Thank you for your reply, I tried by doing this technique:from sklearn.preprocessing import MinMaxScaler……………………….\
 scalar \= MinMaxScaler()\
 normalized \= scalar.fit\_transform(dataset) Do you mean by this?Thank you.    - 
Adrian Tam September 1, 2021 at 11:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-625399 "Direct link to this comment")Yes.        - 
AIbird September 6, 2021 at 8:24 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-625939 "Direct link to this comment")Finally solved the problem. I had issue in my code. I have one more question. The mean accuracy is for training set right?\
 How can I calculate the accuracy in prediction?\
 Can I get some help?            - 
Adrian Tam September 7, 2021 at 6:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-625989 "Direct link to this comment")The accuracy\_metric() function is to do this.292. 
Filip September 19, 2021 at 2:45 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-627145 "Direct link to this comment")This was grat thank you!    - 
Adrian Tam September 19, 2021 at 6:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-627153 "Direct link to this comment")You’re welcomed. Glad you like it.293. 
Franck September 22, 2021 at 8:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-627331 "Direct link to this comment")New to the field : this is exactly what I was looking for! Jason, thanks for this post!I tried to follow it step-by-step and ended-up with 2 questions.Question 1 : updating model doesn’t seem to be so costly, is this because this is a toy program? Does tensorflow (for instance) do more tricky stuffs that make model update costly? On this toy example, this is not easy to understand why model update would be costly.If I get it correctly, here [https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/) you say that mini batch is a tradeoff between SGD and Batch GD and that mini batch is more efficient because model update is done only after it has been evaluated (back propagated). My implementation of mini batch on this toy example would be`` `
~/machinelearningmastery> git diff                                                                                                                                              diff --git a/wheat_seeds.py b/wheat_seeds.py
index 7b78b89..9e7bcd0 100644
--- a/wheat_seeds.py
+++ b/wheat_seeds.py
@@ -141,15 +141,22 @@ def update_weights(network, row, l_rate):
                                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                        neuron['weights'][-1] += l_rate * neuron['delta']`\+def make\_batch(iterable, batch\_size\=1):\
 \+ n \= len(iterable)\
 \+ for i in range(0, n, batch\_size):\
 \+ yield iterable\[i:min(i \+ batch\_size, n)\]\
 \+\
 # Train a network for a fixed number of epochs\
 def train\_network(network, train, l\_rate, n\_epoch, n\_outputs):\
 for epoch in range(n\_epoch):\
 - for row in train:\
 - outputs \= forward\_propagate(network, row)\
 - expected \= \[0 for i in range(n\_outputs)\]\
 - expected\[row\[-1\]\] \= 1\
 - backward\_propagate\_error(network, expected)\
 - update\_weights(network, row, l\_rate)\
 \+ for batch in make\_batch(train, batch\_size\=32):\
 \+ for row in batch: # First backpropagate\
 \+ outputs \= forward\_propagate(network, row)\
 \+ expected \= \[0 for i in range(n\_outputs)\]\
 \+ expected\[row\[-1\]\] \= 1\
 \+ backward\_propagate\_error(network, expected)\
 \+ for row in batch: # Then update model\
 \+ update\_weights(network, row, l\_rate)`` \
 Looks like the exact same cost, no ? Did I miss something ?Question 2: how, when, where the loss function is supposed to be computed in this toy example? For me `sum_error` (in the first version of `train_network`, only used for printing error but not for computation / gradient descent) is useless and is why it disappeared in the final version of `train_network`.For classification, I expected cross entropy to be computed as error for output layer this way`` `
machinelearningmastery> git diff                                                                                                                                              diff --git a/wheat_seeds.py b/wheat_seeds.py
index 7b78b89..0cb0444 100644
--- a/wheat_seeds.py
+++ b/wheat_seeds.py
@@ -3,7 +3,7 @@ from random import seed
 from random import randrange
 from random import random
 from csv import reader
-from math import exp
+from math import exp, log` # Load a CSV file\
 def load\_csv(filename):\
 @@ -111,6 \+111,9 @@ def forward\_propagate(network, row):\
 def transfer\_derivative(output):\
 return output \* (1.0 - output)\+def cross\_entropy(p, q, eps\=1e-15):\
 \+ return -sum(\[p\[i\]\*log(q\[i\]\+eps) for i in range(len(p))\])\
 \+\
 # Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 for i in reversed(range(len(network))):\
 @@ -123,9 \+126,8 @@ def backward\_propagate\_error(network, expected):\
 error \+\= (neuron\['weights'\]\[j\] \* neuron\['delta'\])\
 errors.append(error)\
 else:\
 - for j in range(len(layer)):\
 - neuron \= layer\[j\]\
 - errors.append(expected\[j\] - neuron\['output'\])\
 \+ output \= \[neuron\['output'\] for neuron in layer\]\
 \+ errors.append(cross\_entropy(expected, output))\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\['delta'\] \= errors\[j\] \* transfer\_derivative(neuron\['output'\])\
 `` \
 … But the code breaks and I am not sure to get why?!…Franck    - 
Adrian Tam September 23, 2021 at 3:53 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-627405 "Direct link to this comment")It is too long for me to read it at once but let me answer the first question here. The training is costly because (1) there are many perceptrons to update and (2) there are many data to evaluate. If you consider the simplest gradient descent algorithm, your metric is the MSE function, which involves the entire dataset. If we have M perceptrons and N dataset, there are M weights to train (or more if there are bias terms) and the total number of gradients you need to compute is MxN in each iteration. If your toy example is small in both M and N, you will not notice that is a problem.294. 
Franck October 3, 2021 at 6:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-628461 "Direct link to this comment")Adrian, thanks for the answer: make sense!I tried to use cross entropy as loss function this way :`` `
diff --git a/wheat_seeds.py b/wheat_seeds.py
index 7b78b89..96617d7 100644
--- a/wheat_seeds.py
+++ b/wheat_seeds.py
@@ -111,6 +112,9 @@ def forward_propagate(network, row):
 def transfer_derivative(output):
        return output * (1.0 - output)`\+def cross\_entropy(p, q, eps\=1e-15):\
 \+ return -sum(\[p\[i\]\*log2(q\[i\]\+eps) for i in range(len(p))\])\
 \+\
 # Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 for i in reversed(range(len(network))):\
 @@ -124,8 \+128,9 @@ def backward\_propagate\_error(network, expected):\
 errors.append(error)\
 else:\
 for j in range(len(layer)):\
 - neuron \= layer\[j\]\
 - errors.append(expected\[j\] - neuron\['output'\])\
 \+ neuron\_onehot \= \[0. for neuron in layer\]\
 \+ neuron\_onehot\[j\] \= layer\[j\]\['output'\]\
 \+ errors.append(cross\_entropy(expected, neuron\_onehot))\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\['delta'\] \= errors\[j\] \* transfer\_derivative(neuron\['output'\])\
 ``But classification results are bad : I plotted errors (can’t attach png) and I guess it’s because I’am victim of “vanishing gradient”. If you have any clue or advice, I would be glad to know 😀Franck    - 
Adrian Tam October 6, 2021 at 8:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-628785 "Direct link to this comment")Maybe try with a different activation function. People found that it is the key to mitigate vanishing gradient, but not always works. However, the example here is not deep. The issue of vanishing gradient should not pronounce.295. 
Franck October 21, 2021 at 11:05 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-631017 "Direct link to this comment")Got it to work with a different activation function.At this point, I feel like there is a bug in the code from the post : backprop should start from output with ds/ds \= 1 (as far as I understood it with s \= score) which is not the case unless I am wrong.Am I wrong ?    - 
Adrian Tam October 22, 2021 at 4:13 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-631091 "Direct link to this comment")I understand why you’re thinking like that but that’s meaningless because ds/ds is always 1. We are looking for more interesting subjects such as ds/dw. After all, you can’t change the score. You can only change the weights in the neural network. Hence we prefer to start with ds/dw296. 
SLC October 24, 2021 at 4:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-631615 "Direct link to this comment")Thank you for your code Mr. Jason. However, I am not understanding when you are going to predict the class using the trained network, why and from where are you giving the weights?\
 Thanks in advance.    - 
Adrian Tam October 27, 2021 at 1:44 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-632527 "Direct link to this comment")Weights are in the neural networks. Every neuron is a function y\=f(Wx) where x is the input (usually expressed as a vector of many values) and y is a single value (i.e., scalar). The W is the weight and it is the key thing we need to find during training.297. 
[SD](http://soulkorea.com) November 8, 2021 at 2:37 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-636363 "Direct link to this comment")Hello everyone , I am training a huge dataset which have more than 90 features. Can any one give me some idea to add PCA in the existing code?So that I could reduce my the dimension of my data and use only main features while training.Any help or suggestion would be really appreciated.Thanks.    - 
Adrian Tam November 14, 2021 at 12:08 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-638329 "Direct link to this comment")90 features is not too much, but if you want to use PCA, you can check out this post on dimensionality reduction:\
 [https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/](https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/)298. 
Dyah wardani November 22, 2021 at 3:22 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-640227 "Direct link to this comment")I have a question about the function str\_column\_to\_int(). Why the outputs from 1, 2, 3 change into 2, 0, 1 after use that function?    - 
Adrian Tam November 23, 2021 at 1:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-640562 "Direct link to this comment")I think in this case, your “1”, “2”, “3” are strings and 2, 0, 1 are integers. That’s the result of encoding strings into integers.299. 
Logan January 11, 2022 at 3:46 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-650354 "Direct link to this comment")Here’s just a small correction (I’m sorry for being particular.):In the “3.1. Transfer Derivative” section, you’ve written\
 “Given an output value from a neuron, we need to calculate it’s slope.”It wouldn’t make sense to say\
 “Given an output value from a neuron, we need to calculate it is slope.”\
 (“it is” instead of “it’s”)Therefore, it should be\
 “Given an output value from a neuron, we need to calculate its slope.”\
 (“its” instead of “it’s”)    - 
James Carmichael January 11, 2022 at 8:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-650370 "Direct link to this comment")Thank you for the feedback, Logan!300. 
Durga January 13, 2022 at 8:32 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-650670 "Direct link to this comment")Hi, Mr. Jason thanks for your great tutorial. Can you make me clear about if and else condition in backpropagation error calculation? (if possible please explain these codes in detail)# Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 for i in reversed(range(len(network))):\
 layer \= network\[i\]\
 errors \= list()\
 #— (2) Error computed for the hidden layers: error \= (weight\_k \* error\_j) \* transfer\_derivative(output)\
 if i !\= len(network)-1:\
 for j in range(len(layer)):\
 error \= 0.0\
 #— (A) error \= Sum(delta \* weight linked to this delta)\
 # for each neuron\[LAYER N\+1\] linked to this neuron\[LAYER N\] (current layer)\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 errors.append(error)\
 #— (1) Error computed for the last layer: error \= (expected – output) \* transfer\_derivative(output)\
 else:\
 #— (A) Store the difference between expected and output for each output neuron in errors\[\]\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 errors.append(expected\[j\] – neuron\[‘output’\])\
 # — (B) Store the error signal in delta for each neuron\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])    - 
James Carmichael February 21, 2022 at 2:19 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-655955 "Direct link to this comment")Hi Durga…Please narrow the content of your post down to a single question/comment so that I may better assist you.301. 
Rudra Sonkusare January 27, 2022 at 7:49 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-652241 "Direct link to this comment")Is there any possible way to give a string for input? The string I am trying to give as input is not a meaningful word, for example string \= “zgg7AiPkY37Yvne” and I want to give two of these strings as input to the neural network, any idea how this can be achieved? The current method I use is to convert each character into its decimal code then normalize it in range 0, 1 and thus convert in into a vector of floats.    - 
James Carmichael January 28, 2022 at 10:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-652308 "Direct link to this comment")Hi Rudra…You may find the following of interest:[https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/](https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/)302. 
Andrii February 13, 2022 at 8:51 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-654614 "Direct link to this comment")Hello!\
 I’ve got an issue recently. I’ve implemented back propagation using your approach in C\+\+, however epoch loss doesn’t go done. It may go done with smaller learning rate and bigger number of epochs, but at some point loss goes up to some value again. What can be a potential issue to it? I’ve checked that forward pass and backward pass both work fine.    - 
James Carmichael February 13, 2022 at 12:58 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-654632 "Direct link to this comment")Hi Andrii…While I cannot speak to the C\+\+ implementation, I would recommend the following to move forward with improving your model performance:[https://machinelearningmastery.com/better-deep-learning-neural-networks-crash-course/](https://machinelearningmastery.com/better-deep-learning-neural-networks-crash-course/)303. 
rebot333 February 17, 2022 at 2:02 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-655294 "Direct link to this comment")Thank you so much this is a great lesson    - 
James Carmichael February 18, 2022 at 12:55 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-655460 "Direct link to this comment")You are very welcome! Thank you for the support!304. 
CEN April 3, 2022 at 7:04 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-662827 "Direct link to this comment")hello, Mr. James Carmichael I used the code you created, and it was very useful. can you help me by providing a forecasting plot code for Backpropagation Algorithm with bipolar sigmoid?    - 
James Carmichael April 4, 2022 at 8:59 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-662913 "Direct link to this comment")Hi CEN…Thank you for the feedback! The following resource will be a tremendous help regarding backpropagation.[https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/)305. 
Shraddha April 4, 2022 at 8:03 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-662940 "Direct link to this comment")Thank you, James. The codes were very useful.\
 I tried to implement the above codes in my system. It worked as expected.\
 I was modifying the above code for the MNIST dataset by increasing the number of layers in the existing code. So, can I write this function as Is this the correct way to do it?And one more question, what about the weight update function, do I need to make their changes also?    - 
James Carmichael April 5, 2022 at 7:03 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-663010 "Direct link to this comment")Hi Shraddha…Although I have not executed your code listing, I see no apparent deficiencies. Please let us know what you are specifically trying to accomplish with your code modifications so that we can better assist you.306. 
Shraddha Naik April 5, 2022 at 8:00 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-663084 "Direct link to this comment")Thank you, James Sir. The above code is very useful. I tried to implement the above codes in my system. It worked as expected.\
 But, when I changed the dataset to MNIST, I am getting only 10% accuracy after 1000 epochs. After using mini-batch SGD. Kindly help me with this. Thank You    - 
James Carmichael April 6, 2022 at 8:40 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-663149 "Direct link to this comment")Hi Shraddha…You may find the following helpful:[https://machinelearningmastery.com/optimization-for-machine-learning-crash-course/](https://machinelearningmastery.com/optimization-for-machine-learning-crash-course/)307. 
Shraddha Naik April 5, 2022 at 8:09 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-663086 "Direct link to this comment")def train\_network(network, train, l\_rate, n\_epoch, n\_outputs, kval):\
 for epoch in range(n\_epoch):\
 import random\
 temp \= random.choices(train,k\=kval)\
 for row in temp:\
 outputs \= forward\_propagate(network, row)\
 expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1\
 backward\_propagate\_error(network, expected)\
 update\_weights(network, row, l\_rate)    - 
James Carmichael April 6, 2022 at 8:42 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-663153 "Direct link to this comment")Thank you for the feedback Shraddha!308. 
Rahul May 14, 2022 at 5:22 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-668908 "Direct link to this comment")Hello James,\
 I am working on a project with 9 types of variables and 1 output data. I want to use ANN to get weightage for each variables. I have tried this but I got only an Error in output and the expected data did not find equivalent weights for each individual.\
 Pl, help me.    - 
James Carmichael May 15, 2022 at 10:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669080 "Direct link to this comment")Hi Rahul…Please provide more detail of the nature of the error or errors so that we may better assist you.309. 
nicolas May 18, 2022 at 10:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669603 "Direct link to this comment")Hello James,\
 I am working on a project with 20 entries. But when I change the data packet it gives me errors in the code.\
 Please help me!\
 the data that i tried to put\
 \[\[1539.64, 1006.43, 1549539.885\],\
 \[1537.79, 1004.97, 1545432.816\],\
 \[1535.63, 1003.84, 1541526.819\],\
 \[1533.79, 1002.87645, 1538201.87\],\
 \[1531.65, 1001.80229, 1534410.477\],\
 \[1530.26316, 1000.99, 1531778.121\],\
 \[1528.75778, 1000.46, 1529461.009\],\
 \[1527.07, 999.89813, 1526914.437\],\
 \[1525.76684, 999.40577, 1524860.184\],\
 \[1524.24165, 999.11715, 1522895.973\],\
 \[1523.03339, 998.80306, 1521210.41\],\
 \[1521.88455, 998.56537, 1519701.209\],\
 \[1520.41, 998.26825, 1517777.03\],\
 \[1519.46802, 998.13243, 1516630.307\],\
 \[1518.08149, 997.87776, 1514859.757\],\
 \[1516.89304, 997.7, 1513404.186\],\
 \[1515.94228, 997.6, 1512304.019\],\
 \[1514.99151, 997.48, 1511173.731\],\
 \[1514.15959, 997.32, 1510101.642\],\
 \[1513.24844, 997.1, 1508860.02\],\
 \[1512.32, 996.97637, 1507747.304\]\]    - 
James Carmichael May 19, 2022 at 6:24 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669640 "Direct link to this comment")Hi Nicolas…Please specify what errors you are encountering so that we may better assist you.310. 
nicolas May 19, 2022 at 8:31 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669658 "Direct link to this comment")Hi! James\
 Thank you for your time\
 This is the code that I used\
 And I changed the data but I’m having some errors in it\
 Could you please help me?\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_from math import exp\
 from random import seed\
 from random import random# Calculate neuron activation for an input\
 def activate(weights, inputs):\
 activation \= weights\[-1\]\
 for i in range(len(weights) – 1):\
 activation \+\= weights\[i\] \* inputs\[i\]\
 return activation# Transfer neuron activation\
 def transfer(activation):\
 return 1.0 / (1.0 \+ exp(-activation))# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\
 for layer in network:\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 return inputs# Calculate the derivative of an neuron output\
 def transfer\_derivative(output):\
 return output \* (1.0 – output)# Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 for i in reversed(range(len(network))):\
 layer \= network\[i\]\
 errors \= list()\
 if i !\= len(network) – 1:\
 for j in range(len(layer)):\
 error \= 0.0\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 errors.append(error)\
 else:\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 errors.append(neuron\[‘output’\] – expected\[j\])\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])# Update network weights with error\
 def update\_weights(network, row, l\_rate):\
 for i in range(len(network)):\
 inputs \= row\[:-1\]\
 if i !\= 0:\
 inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]\
 for neuron in network\[i\]:\
 for j in range(len(inputs)):\
 neuron\[‘weights’\]\[j\] -\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] -\= l\_rate \* neuron\[‘delta’\]# Train a network for a fixed number of epochs\
 def train\_network(network, train, l\_rate, n\_epoch, n\_outputs):\
 for epoch in range(n\_epoch):\
 sum\_error \= 0\
 for row in train:\
 outputs \= forward\_propagate(network, row)\
 expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1\
 sum\_error \+\= sum(\[(expected\[i\] – outputs\[i\]) \*\* 2 for i in range(len(expected))\])\
 backward\_propagate\_error(network, expected)\
 update\_weights(network, row, l\_rate)\
 print(‘>epoch\=%d, lrate\=%.3f, error\=%.3f’ % (epoch, l\_rate, sum\_error))# Test training backprop algorithm\
 seed(1)\
 dataset \= \[\[1539.64, 1006.43, 1549539.885\],\
 \[1537.79, 1004.97, 1545432.816\],\
 \[1535.63, 1003.84, 1541526.819\],\
 \[1533.79, 1002.87645, 1538201.87\],\
 \[1531.65, 1001.80229, 1534410.477\],\
 \[1530.26316, 1000.99, 1531778.121\],\
 \[1528.75778, 1000.46, 1529461.009\],\
 \[1527.07, 999.89813, 1526914.437\],\
 \[1525.76684, 999.40577, 1524860.184\],\
 \[1524.24165, 999.11715, 1522895.973\],\
 \[1523.03339, 998.80306, 1521210.41\],\
 \[1521.88455, 998.56537, 1519701.209\],\
 \[1520.41, 998.26825, 1517777.03\],\
 \[1519.46802, 998.13243, 1516630.307\],\
 \[1518.08149, 997.87776, 1514859.757\],\
 \[1516.89304, 997.7, 1513404.186\],\
 \[1515.94228, 997.6, 1512304.019\],\
 \[1514.99151, 997.48, 1511173.731\],\
 \[1514.15959, 997.32, 1510101.642\],\
 \[1513.24844, 997.1, 1508860.02\],\
 \[1512.32, 996.97637, 1507747.304\]\]\
 n\_inputs \= len(dataset\[0\]) – 1\
 n\_outputs \= len(set(\[row\[-1\] for row in dataset\]))\
 network \= initialize\_network(n\_inputs, 2, n\_outputs)\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 for layer in network:\
 print(layer)—————————————————————————————————————————————–Traceback (most recent call last):\
 File “C:\Users\Coder\Downloads\MLP\_v1\_1\1.py”, line 119, in\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 File “C:\Users\Coder\Downloads\MLP\_v1\_1\1.py”, line 86, in train\_network\
 expected\[row\[-1\]\] \= 1\
 TypeError: list indices must be integers or slices, not float    - 
James Carmichael May 20, 2022 at 11:26 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669869 "Direct link to this comment")Hi Nicolas…I do not see any issues from your code listing, however there could be formatting issues related to your code environment that are not readily apparent. Can you try the code in Google Colab?        - 
nicolas May 25, 2022 at 8:02 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-670455 "Direct link to this comment")James Thank you for everything311. 
nicolas May 19, 2022 at 10:47 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-669739 "Direct link to this comment")Hi! James\
 Thank you for your time\
 This is the code that I used\
 And I changed the data but I’m having some errors in it\
 Could you please help me?\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_from math import exp\
 from random import seed\
 from random import random# Calculate neuron activation for an input\
 def activate(weights, inputs):\
 activation \= weights\[-1\]\
 for i in range(len(weights) – 1):\
 activation \+\= weights\[i\] \* inputs\[i\]\
 return activation# Transfer neuron activation\
 def transfer(activation):\
 return 1.0 / (1.0 \+ exp(-activation))# Forward propagate input to a network output\
 def forward\_propagate(network, row):\
 inputs \= row\
 for layer in network:\
 new\_inputs \= \[\]\
 for neuron in layer:\
 activation \= activate(neuron\[‘weights’\], inputs)\
 neuron\[‘output’\] \= transfer(activation)\
 new\_inputs.append(neuron\[‘output’\])\
 inputs \= new\_inputs\
 return inputs# Calculate the derivative of an neuron output\
 def transfer\_derivative(output):\
 return output \* (1.0 – output)# Backpropagate error and store in neurons\
 def backward\_propagate\_error(network, expected):\
 for i in reversed(range(len(network))):\
 layer \= network\[i\]\
 errors \= list()\
 if i !\= len(network) – 1:\
 for j in range(len(layer)):\
 error \= 0.0\
 for neuron in network\[i \+ 1\]:\
 error \+\= (neuron\[‘weights’\]\[j\] \* neuron\[‘delta’\])\
 errors.append(error)\
 else:\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 errors.append(neuron\[‘output’\] – expected\[j\])\
 for j in range(len(layer)):\
 neuron \= layer\[j\]\
 neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])# Update network weights with error\
 def update\_weights(network, row, l\_rate):\
 for i in range(len(network)):\
 inputs \= row\[:-1\]\
 if i !\= 0:\
 inputs \= \[neuron\[‘output’\] for neuron in network\[i – 1\]\]\
 for neuron in network\[i\]:\
 for j in range(len(inputs)):\
 neuron\[‘weights’\]\[j\] -\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] -\= l\_rate \* neuron\[‘delta’\]# Train a network for a fixed number of epochs\
 def train\_network(network, train, l\_rate, n\_epoch, n\_outputs):\
 for epoch in range(n\_epoch):\
 sum\_error \= 0\
 for row in train:\
 outputs \= forward\_propagate(network, row)\
 expected \= \[0 for i in range(n\_outputs)\]\
 expected\[row\[-1\]\] \= 1\
 sum\_error \+\= sum(\[(expected\[i\] – outputs\[i\]) \*\* 2 for i in range(len(expected))\])\
 backward\_propagate\_error(network, expected)\
 update\_weights(network, row, l\_rate)\
 print(‘>epoch\=%d, lrate\=%.3f, error\=%.3f’ % (epoch, l\_rate, sum\_error))# Test training backprop algorithm\
 seed(1)\
 dataset \= \[\[1539.64, 1006.43, 1549539.885\],\
 \[1537.79, 1004.97, 1545432.816\],\
 \[1535.63, 1003.84, 1541526.819\],\
 \[1533.79, 1002.87645, 1538201.87\],\
 \[1531.65, 1001.80229, 1534410.477\],\
 \[1530.26316, 1000.99, 1531778.121\],\
 \[1528.75778, 1000.46, 1529461.009\],\
 \[1527.07, 999.89813, 1526914.437\],\
 \[1525.76684, 999.40577, 1524860.184\],\
 \[1524.24165, 999.11715, 1522895.973\],\
 \[1523.03339, 998.80306, 1521210.41\],\
 \[1521.88455, 998.56537, 1519701.209\],\
 \[1520.41, 998.26825, 1517777.03\],\
 \[1519.46802, 998.13243, 1516630.307\],\
 \[1518.08149, 997.87776, 1514859.757\],\
 \[1516.89304, 997.7, 1513404.186\],\
 \[1515.94228, 997.6, 1512304.019\],\
 \[1514.99151, 997.48, 1511173.731\],\
 \[1514.15959, 997.32, 1510101.642\],\
 \[1513.24844, 997.1, 1508860.02\],\
 \[1512.32, 996.97637, 1507747.304\]\]\
 n\_inputs \= len(dataset\[0\]) – 1\
 n\_outputs \= len(set(\[row\[-1\] for row in dataset\]))\
 network \= initialize\_network(n\_inputs, 2, n\_outputs)\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 for layer in network:\
 print(layer)—————————————————————————————————————————————–Traceback (most recent call last):\
 File “C:\Users\Coder\Downloads\MLP\_v1\_1\1.py”, line 119, in\
 train\_network(network, dataset, 0.5, 20, n\_outputs)\
 File “C:\Users\Coder\Downloads\MLP\_v1\_1\1.py”, line 86, in train\_network\
 expected\[row\[-1\]\] \= 1\
 TypeError: list indices must be integers or slices, not float312. 
wafiq June 1, 2022 at 5:12 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671135 "Direct link to this comment")any code for backpropagation regression? i cant found better than on this page    - 
Adrian Tam June 1, 2022 at 11:30 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671187 "Direct link to this comment")Maybe you can take a look at: [https://machinelearningmastery.com/application-of-differentiations-in-neural-networks/](https://machinelearningmastery.com/application-of-differentiations-in-neural-networks/) \
 For regression, what you need is the activation at **last layer** is a linear function f(x)\=x, so the differentiation f'(x)\=1. Just make this change to the code (either from this post or the code from the link above) and everything else should be just the same.313. 
NOOR AMIRAH June 2, 2022 at 10:29 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671218 "Direct link to this comment")Can I have the train data that uses conjugate gradient method (fletcher-reeves)?314. 
Noor Amirah June 2, 2022 at 10:30 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671219 "Direct link to this comment")Hi, can I have the train data that uses conjugate gradient method (fletcher-reeves)?315. 
Noor Amirah June 2, 2022 at 10:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671220 "Direct link to this comment")I meant did you did you have coding for train dataset that uses fletcher-reeves method?    - 
James Carmichael June 3, 2022 at 9:14 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671294 "Direct link to this comment")Hi Noor…Did you try to implement the code listings that were provided in the tutorial?        - 
Noor Amirah June 8, 2022 at 2:26 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-671658 "Direct link to this comment")Yes, i do but my project is about backpropagation with fletcher-reeves not stochastic..do you have the coding for that?316. 
Eduardo M August 2, 2022 at 6:57 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-678151 "Direct link to this comment")This line in the last for loop in the backpropagation function: neuron\[‘delta’\] \= errors\[j\] \* transfer\_derivative(neuron\[‘output’\])I thought for the last layer (the first layer in the outer for loop), the delta is just actual – expected (what errors\[j\] is equal to). Isn’t the transfer\_derivative term not supposed to be applied in this case ?    - 
James Carmichael August 2, 2022 at 9:02 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-678166 "Direct link to this comment")Hi Eduardo…The following resource is another perspective that may help add clarity:[https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/](https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/)317. 
Confused Coder August 5, 2022 at 3:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-678603 "Direct link to this comment")Why do you add an extra weight to the hidden and output layers?    - 
James Carmichael August 5, 2022 at 9:36 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-679028 "Direct link to this comment")Hello…Please specify the code listing portion you are referring to so that we may better assist you.318. 
Amir Vahedi August 8, 2022 at 6:45 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-679380 "Direct link to this comment")Hi,\
 I have a question:\
 Why you haven’t used any python library such as NumPy and Pandas for this implementation?\
 Why haven’t some nested loops been simplified with the vectorization technique?\
 By doing these I bet the implementation would become more simple and also more efficient.\
 If you didn’t these things on purpose, I am eager to know your reasons.Anyway, this post helped me a lot to understand the implementation behind the neural network, Thank you????    - 
James Carmichael August 9, 2022 at 10:08 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-679445 "Direct link to this comment")Hi Amir…This tutorial and others on our site that are “from scratch” are meant to show you how to code in Python without the libraries so that you may gain understanding and appreciation for libraries such as NumPy and Pandas. After gaining this knowledge you may gain more confidence in utilizing available libraries as opposed to writing the code from scratch.319. 
willow September 21, 2022 at 6:33 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-682186 "Direct link to this comment")Hi, please correct me if I am wrong, but in this example the conventional gradient descent algorithm is used and not stochastic gradient descent, since in the training loop for each training sample the weights are being updated.    - 
James Carmichael September 22, 2022 at 5:25 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-682194 "Direct link to this comment")Hi Willow…You may find the following of interest:[https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/](https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/)320. 
Efemena January 13, 2023 at 11:46 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-688326 "Direct link to this comment")Hi James. Thanks for the tutorial and I’m more grateful for your responses. You are amazing am these years. I am currently working on predicting long term future electricity load demand as a project. I intend to use bpnn in carrying out this forcasting. How do I write a code to use only previous available data in predicting future load demand to say 7 years ahead.321. 
Efemena January 13, 2023 at 11:48 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-688327 "Direct link to this comment")The data available is monthly load demand and some economic factors. My interest is predicting load demand 7 yearsv into the future while having historical monthly load demand and factors as inputs to the neural network    - 
James Carmichael January 14, 2023 at 8:11 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-688329 "Direct link to this comment")Hi Efemena…The following resource is a great starting point:[https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/](https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/)322. 
Hapsoro March 21, 2023 at 10:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-689265 "Direct link to this comment")Hi James. Thanks for the tutorial, I’m so appreciate…but there is one thing that confuses me, can you show for the output of 3 neurons especially for the n\_outputs part.n\_outputs \= len(\[row\[-1\] for row in dataset\])and my dataset\[\[0.21,0.34, 0.65,0, 0 ,1\],\
 \[0.55, 0.67, 0.19, 0, 1, 0\],\
 \[0.77, 0.20, 0.31, 1, 0, 0\]\]Thanks before    - 
James Carmichael March 22, 2023 at 10:03 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-689300 "Direct link to this comment")Hi Hapsoro…Thank you for feedback! Trying to understand your question. Did you execute your code? If so, what were your results?323. 
Kentaro March 29, 2023 at 7:38 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-689477 "Direct link to this comment")Please add indentation for things besides just the functions, as the lack of indentation makes the code very hard to read.    - 
James Carmichael March 30, 2023 at 7:10 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-689522 "Direct link to this comment")Thank you for your feedback and suggestions Kentaro!324. 
Bhaskar September 21, 2023 at 7:42 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-698563 "Direct link to this comment")Hi\
 I need perfect code for Feed Forward Neural Network In r programming\
 please help me    - 
James Carmichael September 22, 2023 at 9:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-698596 "Direct link to this comment")Hi Bhaskar…The following resource is a great starting point:[https://scientistcafe.com/ids/r/ch12dnn](https://scientistcafe.com/ids/r/ch12dnn)325. 
Jim October 27, 2023 at 4:21 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701052 "Direct link to this comment")I get the following error:derivative \= output \* (1 – output)\
 \~\~\^\~\~\~\~\~\~\~\
 TypeError: unsupported operand type(s) for -: ‘int’ and ‘list’How do I make the parameters of the same type?Thanks.326. 
Jim October 27, 2023 at 4:41 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701054 "Direct link to this comment")same with this code:error \= (output – expected) \* transfer\_derivative(output)TypeError: unsupported operand type(s) for -: ‘list’ and ‘list’Thanks.    - 
James Carmichael October 27, 2023 at 9:27 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701075 "Direct link to this comment")Hi Jim…The following discussion may be of interest to you:[https://stackoverflow.com/questions/26685679/typeerror-unsupported-operand-types-for-list-and-list](https://stackoverflow.com/questions/26685679/typeerror-unsupported-operand-types-for-list-and-list)327. 
Michael Roy Ames October 29, 2023 at 10:54 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701207 "Direct link to this comment")Thank you very much for this well written tutorial, Jason. I quite enjoyed figuring it all out, though it took we a couple of weeks to get up-to-speed on the terminology and make it happen.After completing the basic assignment, I updated the code and tried:\
 a) different seeds, learning rates, and epochs\
 b) additional transfer functions: tanh, and gaussian\
 c) multiple hidden layers\
 d) multiple hidden layers of different sizes (numbers of neurons)QUESTION:\
 One thing that got me stuck was the lack of a good visualization tool for viewing the network of layers as they are initialized and trained. I coded a primitive one to troubleshoot and improve my understanding, but there must be something better out there… any suggestions?Now I am looking forward to reading more of your (many!) books – and learning as much as I can.Thanks again.    - 
James Carmichael October 30, 2023 at 8:04 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701249 "Direct link to this comment")Hi Michael…You are very welcome! The following resource may be of interest to you:[https://www.analyticsvidhya.com/blog/2022/03/visualize-deep-learning-models-using-visualkeras/](https://www.analyticsvidhya.com/blog/2022/03/visualize-deep-learning-models-using-visualkeras/)328. 
givonz November 8, 2023 at 5:55 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701812 "Direct link to this comment")You didn’t indent your code. This leaves room for lots of logic errors. Where exactly do your for loops end?329. 
givonz November 8, 2023 at 5:58 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701813 "Direct link to this comment")You should make a note that they need to toggle to code to get the properly formatted python code,.    - 
James Carmichael November 8, 2023 at 10:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-701825 "Direct link to this comment")Thank you for your feedback givonz!330. 
Matthew December 1, 2023 at 12:20 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-703699 "Direct link to this comment")Should the 500 epochs be repeating 5 times?    - 
James Carmichael December 2, 2023 at 11:32 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-703758 "Direct link to this comment")Hi Matthew…The following resource may be of interest to you:[https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)331. 
Tyson December 26, 2024 at 6:35 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-728906 "Direct link to this comment")This is the most straight forward explanation I’ve seen yet. I was able to convert your code into Javascript in one sitting, and I don’t really know Python. Thanks so much!    - 
James Carmichael December 26, 2024 at 8:01 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-728909 "Direct link to this comment")You are very welcome Tyson! We appreciate your feedback and support!332. 
Waleed January 7, 2025 at 5:21 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-729540 "Direct link to this comment")Hi\
 You wrote in the code:\
 errors.append(neuron\[‘output’\] – expected\[j\])but I found in another website the error \= expected – output\
 This will affect new weights differently. Is it related to the way of backpropagation method or there is a mistake? Can you please chick and explain this issue?    - 
James Carmichael January 8, 2025 at 2:20 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-729562 "Direct link to this comment")Hi Waleed…The expression for calculating the error during backpropagation depends on \*\*how the error is defined in the specific implementation\*\*. Both forms you provided—`error = expected[j] - output` and `error = output - expected[j]`—are correct, but they reflect different conventions for defining the error and how the weight updates are derived. Let me explain this in detail.—### \*\*Backpropagation Basics\*\*\
 1. \*\*Error Signal\*\*:\
 – The error signal represents the difference between the predicted output (`output`) and the true target (`expected[j]`).\
 – The sign of this difference affects the direction of the weight adjustment in backpropagation.2. \*\*Gradient Descent Update\*\*:\
 – Weight updates depend on the gradient of the loss function. The gradient tells us the direction in which the weights should be adjusted to minimize the error.\
 – The formula for weight adjustment typically is:\
 \\\[\
 \Delta w \= -\eta \cdot \frac{\partial \text{Loss}}{\partial w}\
 \\\]\
 where \\( \eta \\) is the learning rate.—### \*\*Understanding the Difference\*\*\
 1. \*\*`error = expected[j] - output` (Typical Convention)\*\*:\
 – In this form, the error is defined as the \*\*difference between the target value and the model’s output\*\*.\
 – This is the most common way to define error because it naturally aligns with the goal of minimizing the difference between predictions and targets.\
 – This formulation ensures that:\
 – If the predicted output is too high, the error is negative, and the weight update reduces the prediction.\
 – If the predicted output is too low, the error is positive, and the weight update increases the prediction.2. \*\*`error = output - expected[j]`\*\*:\
 – In this form, the error is defined as the \*\*difference between the model’s output and the target value\*\*.\
 – While less common, this is not inherently wrong. It inverts the sign of the error, which means the weight updates will still adjust correctly \*\*as long as the backpropagation algorithm is consistent with this definition\*\*.—In either case, the algorithm will converge correctly to minimize the loss function, provided the sign convention is applied consistently throughout the backpropagation implementation.—### \*\*Is It a Mistake?\*\*\
 – \*\*No, it’s not necessarily a mistake.\*\* The two forms are just different conventions.\
 – What matters is that the definition of the error matches how the gradients and weight updates are computed.—### \*\*Key Points to Check in Your Code\*\*\
 1. \*\*Consistency in Error Definition\*\*:\
 – If you use `error = expected[j] - output`, ensure the weight updates are based on this error.\
 – If you use `error = output - expected[j]`, verify that the weight updates are consistent with this inverted sign.2. \*\*Loss Function\*\*:\
 – Ensure that the error definition aligns with the loss function you’re using (e.g., Mean Squared Error, Cross-Entropy).3. \*\*Implementation of Gradients\*\*:\
 – Check the formula for computing weight updates to ensure it matches the error definition.—### \*\*Example: Adjusting Weight Updates\*\*\
 Suppose a neuron’s weight update rule is:\
 \\\[\
 w \= w \+ \eta \cdot error \cdot input\
 \\\]\
 If:\
 – `error = expected[j] - output`: Use this directly.\
 – `error = output - expected[j]`: You might need to flip the sign or adjust the gradient computation accordingly.—### \*\*Conclusion\*\*\
 Both `error = expected[j] - output` and `error = output - expected[j]` are valid, but they reflect different conventions for defining error. The important thing is to maintain \*\*consistency\*\* in how the error is defined and used in the backpropagation process.        - 
Waleed January 8, 2025 at 9:59 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-729604 "Direct link to this comment")Thank you, this was very helpfullBased on your discussion, I adjuted this line code to be: errors.append(expected\[j\] – neuron\[‘output’\])then I chaned the “-\=” to “\+\=” in “update network weights with error” part, by adjusting the 2 lines:\
 neuron\[‘weights’\]\[j\] \+\= l\_rate \* neuron\[‘delta’\] \* inputs\[j\]\
 neuron\[‘weights’\]\[-1\] \+\= l\_rate \* neuron\[‘delta’\]I got the same result using the normal equation W(new) \= W(old) \+ delta WThanks & regards333. 
astrologer tejas July 30, 2026 at 6:27 pm [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-771758 "Direct link to this comment")Building a neural network with backpropagation from scratch in Python is a great way to understand how weights, biases, and gradients work together during learning. It provides practical insight into the core concepts behind deep learning without relying on machine learning libraries.    - 
James Carmichael July 31, 2026 at 4:33 am [#](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/#comment-771769 "Direct link to this comment")Thank you for your feedback astrologer!
### Leave a Reply  {#reply-title}

*[October 22, 2021]: 2021-10-22T04:03:43+1100
