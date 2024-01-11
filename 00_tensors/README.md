# Lesson 00: Tensors

Before you start learning about PyTorch and how it builds and trains neural networks, you need to understand how this framework handles data. The core data structure in PyTorch is the **Tensor**. 

In this lesson, you'll learn exactly that: the basics of PyTorch's Tensors, how to create them, how to perform operations on them and how to move them to the GPU for faster computation.

## What is a Tensor?

A Tensor is, essentially, a mathematical concept. It is a generalization of vectors and matrices and it is defined as a multidimensional array of numbers. Thus, a Tensor can be a scalar (a number), a vector (a 1-dimensional array of numbers), a matrix (a 2-dimensional array of numbers) or any n-dimensional array.

Tensors are very useful in the field of Machine Learning and all ML frameworks have their own version of Tensors. Check out the following video by StatQuest, one of the greatest ML-related Youtube channels, to get a grasp of what Tensors are.

[Link to video](https://www.youtube.com/watch?v=L35fFDpwIM4)

After this video, you should have a good understanding of what Tensors are and how they are used in Machine Learning. Now, let's see how PyTorch implements them.

## PyTorch's Tensors

PyTorch stores data in Tensors and every single piece of data in PyTorch is, at its core, a Tensor. This way, everything you'll be doing in PyTorch will involve Tensors in some way or another.

If you've used packages like NumPy before, you'll notice that the Tensor is very similar to NumPy's ndarray. In fact, PyTorch's Tensor is very similar to NumPy's ndarray, but it is tailored to be used in deep learning as it has some extra features that make it more efficient when building neural networks.

Check out Chapter 00 of the [Learn PyTorch for Deep Learning: Zero to Mastery book](https://www.learnpytorch.io/00_pytorch_fundamentals/) and follow along by trying out the code in the [notebook in this folder](./tensors_101.ipynb).

After this lesson, you should have a good understanding of how to use PyTorch's Tensors and how to perform operations on them.

Now, you can move on to the next lesson, where you'll learn about the PyTorch workflow and how to build a simple linear regression model. After that lesson, there's going to be an <ins>assignment</ins> where you can apply what you've learned in the first two lessons.

## Extra Resources

If you want to deepen your knowledge on Tensors, check out the [PyTorch documentation](https://pytorch.org/docs/stable/tensors.html).

Also, you're more than welcome to do the exercises mentioned at the end of the lesson. Just fill in the code in the [template notebook](./00_pytorch_fundamentals_exercises.ipynb) and check their solutions afterwards.

Our blog post about this is [here on the DareData Blog](https://blog.daredata.engineering/pytorch-introduction-the-tensor-object/)