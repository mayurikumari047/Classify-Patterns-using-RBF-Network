# Classify-Patterns-using-RBF-Network
Design and implementation of an Radial Basis Function (RBF) network and classify non linear patterns based on RBF network

When a radial-basis function (RBF) network is used to perform a complex pattern classification
task, the problem is basically solved by first transforming it into a high dimensional space in a nonlinear manner and then separating the classes in the output layer. 

### Cover’s Theorem on the separability of patterns:
A complex pattern-classification problem, cast in a high-dimensional space nonlinearly,
is more likely to be linearly separable than in a low-dimensional space, provided that the
space is not densely populated.

As per the Cover’s theorem, it’s clear that mapping input patterns nonlinearly in a high dimension space in hidden layer is more likely to classify the input patterns than in lower dimension space.
Here in this project, when I used 20 centers (higher dimensional mapping) for calculation of output using RBF, I achieved 100% accuracy and I was able to classify both classes properly. Decision boundary drawn in this case was clearly classifying both the classes. 
Whereas, when I used only 4 centers (lower dimension mapping) for calculation of output using RBF, achieved only 80% accuracy and classification was not proper as compared to earlier case of 20 centers.


#### Input details, hyper-parameter values and function used

Input size: 100

Numbers of centers taken: 20 and 4 separately

Input range: 0 to 1

Initial Weight range: -1 to 1

Learning parameter, eta = 1

Resolution used for plotting decision boundary: 1000

Radial Basis function used: Gaussian function


#### Color legends used in the graph:

Red Cross: Class plus inputs

Blue Cross: Class minus inputs

Red Dots: Class plus initial center

Black Dots: Class plus final centers

Blue Dots: Class minus initial centers

Magenta Dots: Class minus final centers
