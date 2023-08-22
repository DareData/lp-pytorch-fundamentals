# Lesson 01: PyTorch Workflow and Linear Regression

Before you start building neural networks, you need to understand how PyTorch works and how to build a simple model. In this lesson, you'll learn how to use PyTorch to build a simple linear regression model.

## Linear Regression...?

If you don't have a mathematics background and never heard of linear regression, or even if you just need a refresher, this section is for you. Linear Regression is nothing more than just fitting a line that best describes a set of data points. This can be solved analytically, for a few points, but Machine Learning allows us to slove problems with a lot more data. Check ou this short video and you'll be up to speed with regression in just 2 minutes.

[Link to video](https://www.youtube.com/watch?v=CtsRRUddV2s)

If you're curious and want to know how exactly we fit a line to a set of points and how we evaluate what is a good fit and what's not, check out this video by StatQuest that briefly delves into that. 

[Link to video](https://www.youtube.com/watch?v=PaFPbb66DxQ)

However, if you wish to really get into the mathematics behind linear regression, this longer video by StatQuest is a great resource and will take you through the mathematical journey of linear regression.

[Link to video](https://www.youtube.com/watch?v=nk2CQITm_eo)

## PyTorch Workflow

Almost all PyTorch projects follow the same workflow. You'll need to:
1. Prepare the data (create Tensors)
2. Build a model through class inheritance
3. Train the model with a training loop
4. Analyze the model's results
5. Save the model

This workflow is very similar to the one you'll find in other frameworks, such as TensorFlow, and once you've understood it, you'll be able to replicate it to build any model you want, although some model types may require some extra steps.

Check out Chapter 01 of the [Learn PyTorch for Deep Learning: Zero to Mastery book](https://www.learnpytorch.io/01_pytorch_basics/) and follow along by trying out the code in the [notebook in this folder](./pytorch_workflow.ipynb).

After this lesson, you'll be able to build a simple linear regression model in PyTorch and understand how to use the PyTorch workflow to build any model you want.


## Assignment 1

Having completed these two lessons, you're now ready to do **Assignment 1**. In this assignment, you'll build a linear regression model to predict the sales of a company based on its advertising budget. You'll also learn how to visualize the results of your model and how to improve it. The assignment is in the `assignments/assignment_1` folder. Good luck!

## Extra Resources

Once again, you are encouraged to do the exercises shown in the chapter of this book. These can help you prepare for the assignment, but also to get a better understanding of the concepts. Feel free to do the exercises in the template [provided by the book](./01_pytorch_workflow_exercises.ipynb) and check out the solutions as well.

If you want to see an implementation of a linear regression model in PyTorch from scratch and maybe a few extra details that were not covered in the book, check out this great [Kaggle article](https://www.kaggle.com/code/aakashns/pytorch-basics-linear-regression-from-scratch) by Aakash N S.

Also, check out [PyTorch's Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) for a quick reference on how to use PyTorch.

