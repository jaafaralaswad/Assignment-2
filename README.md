![Python Version](https://img.shields.io/badge/python-3.12-blue)
![OS](https://img.shields.io/badge/os-ubuntu%20%7C%20macos%20%7C%20windows-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[![codecov](https://codecov.io/gh/jaafaralaswad/Assignment-1/branch/main/graph/badge.svg)](https://codecov.io/gh/jaafaralaswad/Assignment-1) ![GitHub Actions](https://github.com/jaafaralaswad/Assignment-1/actions/workflows/tests.yml/badge.svg)


# ME700 Assignment 1

## Table of Contents

- [Introduction](#introduction)
- [Conda Environment, Installation, and Testing](#conda-environment-installation-and-testing)
- [The Bisection Method](#the-bisection-method)
- [Newton's Method](#newtons-method)
- [Elastoplasticity](#elastoplasticity)
- [Tutorials](#tutorials)
- [More Information](#more-information)

## Introduction
This repository presents the work developed to fulfill the requirements of Assignment 1 for the course ME700. It involves codes to solve algebraic and mechanical problems using the bisection and Newton-Raphson methods. Also, it contains a code to solve geometrically linear 1D elastoplasticity problems with isotropic and kinematic hardening using the predictor-corrector algorithm.


## Conda environment, install, and testing

This procedure is very similar to what we did in class. First, you need to download the repository and unzip it. Then, to install the package, use:

```bash
conda create --name assignment-1-env python=3.12
```

After creating the environment (it might have already been created by you earlier), make sure to activate it, use:

```bash
conda activate assignment-1-env
```

Check that you have Python 3.12 in the environment. To do so, use:

```bash
python --version
```

Create an editable install of the assignemnt codes. Use the following line making sure you are in the correct directory:

```bash
pip install -e .
```

You must do this in the correct directory; in order to make sure, replace the dot at the end by the directory of the folder "Assignment-1-main" that you unzipped earlier: For example, on my computer, the line would appear as follows:

```bash
pip install -e /Users/jaafaralaswad/Downloads/Assignment-1-main
```

Now, you can test the code, make sure you are in the tests directory. You can know in which directory you are using:

```bash
pwd
```

Navigate to the tests folder using the command:

```bash
cd
```

On my computer, to be in the tests folder, I would use:

```bash
cd /Users/jaafaralaswad/Downloads/Assignment-1-main/tests
```


Once you are in the tests directory, use the following to run the tests:

```bash
pytest -s test_main.py
```

Code coverage should be 100%.

To run the tutorial, make sure you are in the tutorials directory. You can navigate their as you navigated to the tests folder. On my computer, I would use:

```bash
cd /Users/jaafaralaswad/Downloads/Assignment-1-main/tutorials
```

Once you are there, you can use:

```bash
pip install jupyter
```

Depending on which tutorial you want to use, you should run one of the following lines:

```bash
jupyter notebook bisection.ipynb
```

```bash
jupyter notebook newton.ipynb
```

```bash
jupyter notebook elastoplasticity.ipynb
```

A Jupyter notebook will pop up, containing five numerical examples.



## The Bisection Method

The bisection method is a classical numerical technique for finding real roots of algebraic equations. It is based on the intermediate value theorem from calculus, which states that if a continuous function takes on opposite signs at two points, there must be at least one root between them. The method's name reflects how it works: in each iteration, the interval is bisected, and the half containing a sign change is retained for the next iteration, while the other half is discarded.

The user must define the function, $f(x)$, and specify the lower and upper bounds of the interval, $a$ and $b$, respectively. To ensure the presence of a root, the function values at these bounds must have opposite signs. If both values share the same sign, the existence of a root within the interval is not guaranteed. In such cases, an error message is displayed, prompting the user to select a different set of boundaries.

The bisection method is an iterative process, and the user determines the termination criteria based on the required accuracy, which depends on the specific application. This solver employs two termination criteria, ending the iterations when either is satisfied. The first criterion is when $|c-a|< \epsilon_1$, meaning the half-interval size becomes smaller than a predefined threshold. The second criterion is when $|f(c)|< \epsilon_2$, indicating that the function value is sufficiently close to zero. The user must specify both $\epsilon_1$ and $\epsilon_2$, with tighter tolerances providing greater accuracy at the cost of additional iterations.

The concept of the bisection method is straightforward. However, the method has significant limitations. First, the user must identify an interval that contains a root.  Second, the method can only find a single root at a time; for equations with multiple roots, the user must test different intervals to locate each one. Third, the method has a relatively slow convergence rate, often requiring more iterations compared to more advanced numerical techniques.


## Newton's Method

Newton's method is an efficient numerical technique for finding real roots of equations. It iteratively refines an initial guess, $x_0$, using the formula:  

$$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $$

The method stops when either $|x_{n+1} - x_n| < \epsilon_1$ or  $|f(x_n)| < \epsilon_2$, ensuring sufficient accuracy.  

Unlike the bisection method, Newton-Raphson converges quadratically when $x_0$ is close to the root. However, it requires $f'(x)$, may fail if $f'(x) = 0$, and can diverge from poor initial guesses.

For systems of equations  $\mathbf{F}(\mathbf{x}) = 0$, the method extends to multiple dimensions using the Jacobian matrix $\mathbf{J}$:  

$$ \mathbf{x}_{n+1} = \mathbf{x}_n - \mathbf{J}^{-1} \mathbf{F}(\mathbf{x}_n)$$

Here, $\mathbf{J}$ is the matrix of partial derivatives $\frac{\partial F_i}{\partial x_j}$.


## Elastoplasticity

To be written.


## Tutorials

This repository contains three tutorials. Each contains five numerical examples addressing one of the topics mentioned above.

## More information

More information can be found here:

- [Bisection Method](https://en.wikipedia.org/wiki/Bisection_method)
- [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method)
- [Plasticity](https://en.wikipedia.org/wiki/Plasticity_(physics))

