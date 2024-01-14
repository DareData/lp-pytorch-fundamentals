# Lesson 02: Neural Networks and Classification

In this lesson, you'll learn how to build a neural network to classify two different classes of points. Once again, this lesson follows the same workflow as the previous one, but this time you'll be building a neural network instead of a linear regression model and you'll get to understand how neural networks work while building one with PyTorch.

## Neural Networks

Neural networks are the most popular type of model in Deep Learning. They are very powerful and can be used to solve a wide variety of problems. 

In an abstract way, neural networks are a software imitation attempt of the human brain, hence *neural*. They are composed of neurons, which are connected to each other and have a certain weight associated with each connection. These weights are what the model learns during training and are used to make predictions. 

In practice though, neural networks are a series of matrix multiplications and non-linear functions with the goal of approximating a function that maps the input to the output by minimizing a loss function.

Watch the following short video about Neural Networks to visually understand the basics of their structure and how they work.

[Link to video](https://www.youtube.com/watch?v=bfmFfD2RIcg)

If you wish to delve deeper into the mathematics behind neural networks, check out this video by 3Blue1Brown. It's a bit longer, but it's truly worth it as he's a great educator and will show you the intuition behind the math.

[Link to video](https://www.youtube.com/watch?v=aircAruvnKk)

## Classification with PyTorch

Classification is a type of problem where we want to predict a class for a given input. For example, we might want to predict if a person has a disease or not based on some medical tests, or if a person will like a movie or not based on their previous ratings. This problem is different from regression because we're not trying to predict a continuous value, but rather a discrete one, like a category. In this lesson, we'll start with binary classification, i.e., the output will be one of two classes, but this is usually extended to more classes.

Your challenge for this lesson is, once again, to check out Chapter 02 of the [Learn PyTorch for Deep Learning: Zero to Mastery book](https://www.learnpytorch.io/02_neural_networks/) and follow along by trying out the code in the [notebook in this folder](./neural_networks.ipynb).

After this lesson, you'll be able to build a simple neural network in PyTorch and understand how to use the PyTorch to create any neural network you want.

When you're finished, you can move on to the next lesson. At the end of the lesson, you'll be able to do **Assignment 2**, which will focus on the concepts you've learned in this lesson and the next one.

## Extra Resources

As previously recommended, try to do the exercises shown at the end of the chapter, which you can complete by filling in the template provided by the book [here](./02_neural_networks_exercises.ipynb). You can also check out the solutions when you're done.

Also, check out [Tensorflow's Playground](https://playground.tensorflow.org/), where you can visually see how neural networks work and how they learn. It's a great resource to visually get a better understanding of neural networks and it's not only fun to play around with, but also spectacular to watch a mathematical algorithm learn.

Now that you mostly know how to use PyTorch to build something functional, take a look at the [Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html) to get a better understanding of the concepts you've learned so far. You can also use it as a reference for the rest of the course.

At last, earlier we mentioned a 3Blue1Brown video about neural networks. In fact, has a whole series of videos about neural networks and Deep Learning, which are a great resource to learn more if you care about the mathematics that make neural networks work. He's a great educator and the visual explanations will unlock a whole new perspective about NNs. You can check out the whole series [here](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).

Our blog post about the topic of Non-Linear activation functions can be found [here](https://blog.daredata.engineering/pytorch-introduction-enter-nonlinear-functions/)