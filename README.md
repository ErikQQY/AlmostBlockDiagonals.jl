# AlmostBlockDiagonals

[![Build Status](https://github.com/ErikQQY/AlmostBlockDiagonals.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ErikQQY/AlmostBlockDiagonals.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Pivoting factorization and linear solver for almost block diagonal matrices. Mainly designed for linear solving in BVP. Users who only want to use this package should pay attention.

```julia
a1 = [ 0.1   2.0  -0.1  -0.1
0.2  -0.2  -0.2   4.0
-1.0   0.3  -0.3   0.3]
a2 = [-0.4  0.4  -5.0
 3.0  0.5  -0.5]
a3 = [0.6  -0.6  -0.6   5.0
0.5   4.0   0.5  -0.5
3.0   0.4  -0.4   0.4]
a4 = [0.3  -0.3  0.3  7.0]
a5 = [0.2  -0.2  -0.2   8.0
6.0   0.1  -0.1  -0.1]
A = AlmostBlockDiagonal([a1, a2, a3, a4, a5], [2, 3, 1, 1, 4])
B = [1.94,3.04,-0.83, -3.54,2.75,1.32,2.35,1.96,1.52,0.78,2.40]
x = A\B
```
```julia
11-element Vector{Float64}:
 1.0999999999999999
 1.0
 0.9
 0.8
 0.7
 0.6
 0.5000000000000001
 0.39999999999999997
 0.30000000000000004
 0.2
 0.1
```

For details about algorithms, please see:
[SOLVEBLOK: A Package for Solving Almost Block Diagonal Linear Systems](https://dl.acm.org/doi/pdf/10.1145/355873.355880)