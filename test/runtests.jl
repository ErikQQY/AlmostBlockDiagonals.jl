using AlmostBlockDiagonals
using Test

@testset "Test Linear Solve in AlmostBlockDiagonals.jl" begin
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
    @test isapprox(x, [ 1.0999999999999999, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5000000000000001, 0.39999999999999997, 0.30000000000000004, 0.2,  0.1])
end

@testset "Test case in F01LHF" begin
    a1 = [-1.00 -0.98 -0.79 -0.15
        -1.00  0.25 -0.87  0.35]

    a2 = [0.78  0.31 -0.85  0.89 -0.69 -0.98 -0.76
        -0.82  0.12 -0.01  0.75  0.32 -1.00 -0.53
        -0.83 -0.98 -0.58  0.04  0.87  0.38 -1.00
        -0.21 -0.93 -0.84  0.37 -0.94 -0.96 -1.00]

    a3 = [-0.99 -0.91 -0.28  0.90  0.78 -0.93 -0.76  0.48
        -0.87 -0.14 -1.00 -0.59 -0.99  0.21 -0.73 -0.48
        -0.93 -0.91  0.10 -0.89 -0.68 -0.09 -0.58 -0.21
        0.85 -0.39  0.79 -0.71  0.39 -0.99 -0.12 -0.75
        0.17 -1.37  1.29 -1.59  1.10 -1.63 -1.01 -0.27]

    a4 = [0.08  0.61  0.54 -0.41  0.16 -0.46
        -0.67  0.56 -0.99  0.16 -0.16  0.98
        -0.24 -0.41  0.40 -0.93  0.70  0.43]

    a5 = [0.71 -0.97 -0.60 -0.30  0.18
        -0.47 -0.98 -0.73  0.07  0.04
        -0.25 -0.92 -0.52 -0.46 -0.58
        0.89 -0.94 -0.54 -1.00 -0.36]

    B = [-2.92, -1.27, -1.30, -1.17, -2.10, -4.51, -1.71, -4.59,
        -4.19, -0.93, -3.31, 0.52, -0.12, -0.05, -0.98, -2.07,
        -2.73, -1.95]
    A = AlmostBlockDiagonal([a1, a2, a3, a4, a5], [1, 3, 6, 3, 5])
    x = A\B

    @test isapprox(x, ones(18))
end

@testset "Yet another test case" begin
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
    B = [1.9, 3.8, -0.7, -5.0, 3.0, 4.4, 4.5, 3.4, 7.3, 7.8, 5.9]
    A = AlmostBlockDiagonal([a1, a2, a3, a4, a5], [2, 3, 1, 1, 4])
    x = A\B

    @test isapprox(x, ones(11))
end

@testset "Another test case where last[1]=0" begin
    # in this case, the almost block diagonal matrix is in the structure of like:
    #               TOPBLK
    #               ARRAY(1)
    #                     ARRAY(2)
    #                          .
    #                             .
    #                                .
    #                                   .
    #                                    ARRAY(NBLOKS)
    #                                           BOTBLK
    a1 = [-1.00 -0.98 -0.79 -0.15
        -1.00  0.25 -0.87  0.35]

    a2 = [0.78  0.31 -0.85  0.89 -0.69 -0.98 -0.76
        -0.82  0.12 -0.01  0.75  0.32 -1.00 -0.53
        -0.83 -0.98 -0.58  0.04  0.87  0.38 -1.00
        -0.21 -0.93 -0.84  0.37 -0.94 -0.96 -1.00]

    a3 = [-0.99 -0.91 -0.28  0.90  0.78 -0.93 -0.76  0.48
        -0.87 -0.14 -1.00 -0.59 -0.99  0.21 -0.73 -0.48
        -0.93 -0.91  0.10 -0.89 -0.68 -0.09 -0.58 -0.21
        0.85 -0.39  0.79 -0.71  0.39 -0.99 -0.12 -0.75
        0.17 -1.37  1.29 -1.59  1.10 -1.63 -1.01 -0.27]

    a4 = [0.08  0.61  0.54 -0.41  0.16 -0.46
        -0.67  0.56 -0.99  0.16 -0.16  0.98
        -0.24 -0.41  0.40 -0.93  0.70  0.43]

    a5 = [0.71 -0.97 -0.60 -0.30  0.18 0.78
        -0.47 -0.98 -0.73  0.07  0.04  0.15
        -0.25 -0.92 -0.52 -0.46 -0.58  0.46
        0.89 -0.94 -0.54 -1.00 -0.36  0.27]

    C = [-5.84, -2.54, -2.60, -2.34, -4.20, -9.02, -3.42, -9.18, -8.38, -1.86, -6.62, 1.04, -0.24, -0.1, -0.40, -3.84, -4.54, -3.36]

    B = [-2.92, -1.27, -1.30, -1.17, -2.10, -4.51, -1.71, -4.59,
        -4.19, -0.93, -3.31, 0.52, -0.12, -0.05, -0.2, -1.92,
        -2.27, -1.68]
    A = AlmostBlockDiagonal([a1, a2, a3, a4, a5], [0, 3, 6, 3, 6])
    x = A\B

    @test isapprox(x, ones(18))
end

@testset "Another test case where last[end]=0" begin
    # in this case, the almost block diagonal matrix is in the structure of like:
    #               TOPBLK
    #               ARRAY(1)
    #                     ARRAY(2)
    #                          .
    #                             .
    #                                .
    #                                   .
    #                                    ARRAY(NBLOKS)
    #                                           BOTBLK
    a1 = [-1.00 -0.98 -0.79 -0.15
        -1.00  0.25 -0.87  0.35]

    a2 = [0.78  0.31 -0.85  0.89 -0.69 -0.98 -0.76
        -0.82  0.12 -0.01  0.75  0.32 -1.00 -0.53
        -0.83 -0.98 -0.58  0.04  0.87  0.38 -1.00
        -0.21 -0.93 -0.84  0.37 -0.94 -0.96 -1.00]

    a3 = [-0.99 -0.91 -0.28  0.90  0.78 -0.93 -0.76  0.48
        -0.87 -0.14 -1.00 -0.59 -0.99  0.21 -0.73 -0.48
        -0.93 -0.91  0.10 -0.89 -0.68 -0.09 -0.58 -0.21
        0.85 -0.39  0.79 -0.71  0.39 -0.99 -0.12 -0.75
        0.17 -1.37  1.29 -1.59  1.10 -1.63 -1.01 -0.27]

    a4 = [0.08  0.61  0.54 -0.41  0.16 -0.46
        -0.67  0.56 -0.99  0.16 -0.16  0.98
        -0.24 -0.41  0.40 -0.93  0.70  0.43]

    a5 = [0.71 -0.97 -0.60]


    B = [-2.92, -1.27, -1.30, -1.17, -2.10, -4.51, -1.71, -4.59, -4.19, -0.93, -3.31, 0.52, -0.12, -0.05, -0.86]
    A = AlmostBlockDiagonal([a1, a2, a3, a4, a5], [0, 3, 6, 3, 3])
    x = A\B

    @test isapprox(x, ones(15))
end

@testset "Test on factorization and solving on IntermediateAlmostBlockDiagonal" begin
    a1 = [  0.1   2.0  -0.1  -0.1
    0.2  -0.2  -0.2   4.0
   -1.0   0.3  -0.3   0.3]
    a2 = [  0.0  0.0   0.0
    -0.4  0.4  -5.0
     3.0  0.5  -0.5]
    a3 = [0.6  -0.6  -0.6   5.0
    0.5   4.0   0.5  -0.5
    3.0   0.4  -0.4   0.4]
    a4 = [ 0.0   0.0  0.0  0.0
    0.0   0.0  0.0  0.0
    0.3  -0.3  0.3  7.0]
    a5 = [ 0.0   0.0   0.0   0.0
    0.0   0.0   0.0   0.0
    0.2  -0.2  -0.2   8.0
    6.0   0.1  -0.1  -0.1]
    B= [1.94,3.04,-0.83, -3.54,2.75,  1.32,2.35,1.96, 1.52,0.78,2.40]
    IA = IntermediateAlmostBlockDiagonal([a1, a2, a3, a4, a5], [2, 3, 1, 1, 4])
    ipivot = zeros(Integer, 16)
    scrtch = zeros(16)
    factor_shift(IA, ipivot, scrtch)
    substitution(IA, ipivot, B) # directly inplace computing

    @test isapprox(B, [ 1.0999999999999999, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5000000000000001, 0.39999999999999997, 0.30000000000000004, 0.2,  0.1])
end