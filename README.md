## TUTORIAL: Semidefinite Programming for Intensity Only Estimation of the Transmission Matrix

The possibility of measuring the transmission matrix using intensity-only measurements is a much sought-after feature as it allows us not to rely on interferometry. Interferometry usually requires laboratory-grade stability difficult to obtain for real-world applications. Typically, we want to be able to retrieve the transmission matrix from a set of pairs composed of input masks and output intensity patterns. However, this problem, which corresponds to a phase-retrieval problem, is not convex, hence difficult to solve using standard techniques. The idea proposed in [I. Waldspurger *et al.*, Math. Program (2015)](https://doi.org/10.1007/s10107-013-0738-9) is to relax some constraints to approximate the problem to a convex one that can be solved using the semidefinite programming approach. I briefly detail the approach and provide an example of the procedure to reconstruct the transmission matrix using Python. A Jupyter notebook can be found on my Github account: [semidefiniteTM_example.ipynb](https://github.com/wavefrontshaping/WFS.net/blob/master/semidefiniteTM_example.ipynb).

## Context

Measuring the full complex transmission matrix requires having access to the phase of the optical field. While there exist non-interferometric approaches, they usually reduce the resolution, which is detrimental for complex media applications where the speckle pattern shows high spatial frequency fluctuations. Other methods require measuring the intensity pattern at different planes, adding more constraints to the experimental setup. Ideally, we want to be able to reconstruct the transmission matrix from a set of pairs, consisting of one input field and the corresponding output intensity pattern. Various approaches were proposed, in particular statistical machine learning, deep learning, and semidefinite programming. We will focus here on the semidefinite programming approach.

This approach was first proposed in [I. Waldspurger *et al.*, Math. Program (2015)](https://doi.org/10.1007/s10107-013-0738-9) and was later demonstrated for the first time in [N'Gom et al., Sci. Rep. (2017)](https://doi.org/10.1038/s41598-017-02716-x) to measure the transmission matrix of a scattering medium.

## The mathematical problem

Let's consider a linear medium of transmission matrix $\mathbf{H}$ of size $M\times N$ that links the input field $x$ to the output one $y$. The $j^\text{th}$ line of the transmission matrix $H_i$ corresponds to the effect of the different input elements on the $j^\text{th}$ output measurement point of the field $y_j$. The reconstruction of each line of the matrix can be treated independently, we consider only the output pixel $j$ in the following.

We consider that we have at our disposal a set of input/output pairs $\left{X^k,\lvert Y_j^k \rvert \right}$, with $k \in [1...P]$, where $X_k$ is a complex vector corresponding to an input wavefront, $Y_j^k=\mathbf{H}X^k= \lvert Y_j^k\rvert \exp^{i\Phi_k}$ is the corresponding output complex field and $P$ is the number of elements in the data set. $\mathbf{X}$ is the matrix containing all the input training masks, and $Y_j$ is the vector containing the output fields at the target point $j$ for all input masks.

As we only have access to the amplitude $\lvert Y_j\rvert$ of the output field, we want to solve:

```math
\begin{aligned}
    \text{min.} & \quad \lVert H_j\mathbf{X}-\lvert Y_j\rvert\exp^{i\Phi_j}\rVert_ 2^2 \\
    \text{subject to} & \quad H_j \in \mathbb{C}^M, \, \Phi_j \in [0,2\pi]^P
\end{aligned}
```

It is shown in [I. Waldspurger *et al.*, Math. Program (2015)](https://doi.org/10.1007/s10107-013-0738-9) that this expression can be simply rearranged to become:

```math
\begin{aligned}
    \text{min.} & \quad u^\dagger \mathbf{Q} u = \mathrm{Tr}\left(\mathbf{Q}u u^\dagger\right) \\
    \text{subject to} & \quad H_j \in \mathbb{C}^M, \,u\in\mathbb{C}^P,\,\lvert u_k\rvert=1 \quad \forall k\in[0..P]
\end{aligned}
```

with $\mathbf{Q} = \text{diag}(\lvert Y_j\rvert)\left(\mathbf{I}-\mathbf{X}\mathbf{X}^p\right) \text{diag}(\lvert Y_j\rvert)$.

$^p$ stands for the Moore-Penrose pseudoinverse and $\dagger$ for the transpose conjugate. The vector $u$ contains the phase of the $j^\text{th}$ output point for all the elements of the data set, so that $u_k=\exp^{i\Phi_k}$. The equivalence between these two expressions is guaranteed by the fact that $\mathbf{Q}$ is a positive semidefinite Hermitian matrix.

By construction, $\mathbf{U}=u_j u_j^\dagger$ is of rank equal to $1$. By relaxing this constraint, this problem can be written as a convex problem that can be solved using semidefinite programming:

```math
\begin{aligned}
    \text{min.} & \quad \mathrm{Tr}\left(\mathbf{Q}\mathbf{U}\right) \\
    \text{subject to} & \quad \mathbf{U}=\mathbf{U}^\dagger,\, \text{diag}\left(\mathbf{U}\right) = 1, \mathbf{U} \succeq 0
\end{aligned}
```

with $\mathbf{U} \succeq 0$ denoting the positive semidefinite constraint on $\mathbf{U}$. We can now use standard convex solvers to find a solution. The difficulty is that $\mathbf{U}$ is not of rank $1$ anymore. To find an approximate solution, we take the first singular vector $V_0$ of $\mathbf{U}$ which gives the phase of the output field with good accuracy.

Now that we have the complex output field, we can use a pseudo-inversion to retrieve the transmission matrix.

```math
H_j = \lvert Y \rvert V_0\mathbf{X}^p
```

## Python implementation

The only important part concerns solving the convex problem. In **Python**, [CVXPY](https://www.cvxpy.org/) allows writing the problem in a natural way, i.e. exactly as we wrote it in equation (3). The **Matlab** module [CVX](http://cvxr.com/cvx/) does the same thing.

The part of the code that corresponds to solving the convex problem is very concise:

```python
Q = np.diag(np.abs(Y))@(np.eye(p)-np.linalg.pinv(X)@X)@np.diag(np.abs(Y))
# Variable (unknown) of the problem
U = Variable((p,p), hermitian=True)
# Objective to minimize
objective = Minimize(abs(trace(Q@U)))
# Constraints
# 1. U is positive semidefinite
constraints = [U >> 0]
# 2. Elements of the diagonal are 1.
for k in range(p):
    constraints += [U[k, k] == 1]
prob = Problem(objective, constraints)
# 3. U is Hermitian (already set at the initialization of U)
# Run the solver
prob.solve(solver=SCS, verbose=True, eps=1e-5, max_iters=100000)
```

## Full example
A full Python code that simulates the reconstruction of a random transmission matrix using this procedure in the presence of noise can be found [here](./semidefiniteTM_example.ipynb).

 
## Remarks

Using this approach, which is also the case when using machine learning, the output pixels are treated independently. For each output pixel, the system is not sensitive to a global phase shift or conjugation. That implies that the relative phase between the lines of the matrix is not known. That is not detrimental for the generation of output intensity patterns, but can be otherwise important. It would then require an additional measurement to find these relative phases.
