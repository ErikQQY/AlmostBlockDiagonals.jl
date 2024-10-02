module AlmostBlockDiagonals

using ConcreteStructs
import Base.\

"""
    AlmostBlockDiagonal{T, I <: Integer, V <: AbstractMatrix{T}} <: AbstractMatrix{T}

A matrix with block matrices on the diagonal, but not strictly has their corner against each other.

For example:

A 10 x 10 matrix:

x  x  x  x
x  x  x  x
x  x  x  x
      x  x  x  x
      x  x  x  x
            x  x  x  x
            x  x  x  x
                  x  x  x  x
                  x  x  x  x
                  x  x  x  x

can be abstracted as:

```julia
julia> AlmostBlockDiagonal([rand(3,4),rand(2,4),rand(2,4),rand(3,4)], [2,2,2,4])
```

Here, the first argument is the fillers in the almost block diagonal matrix, the second argument is the offset of each adjacent block in the diagonal.

! note
    The column of block `ncol` and row of block `nrow` must satisfy: `ncol` ≥ `nrow`.

The implementation mainly comes from the paper: [SOLVEBLOK: A Package for Solving Almost Block Diagonal Linear Systems](https://dl.acm.org/doi/pdf/10.1145/355873.355880)
which is originally a FORTRAN program for the solution of an almost block diagonal system by gaussian elimination with scale row pivoting.
"""
@concrete struct AlmostBlockDiagonal{T, I <: Integer, V <: AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    lasts::Vector{I}
    rows::Vector{I}
    cols::Vector{I}

    function AlmostBlockDiagonal{T, I, V}(blocks::Vector{V}, lasts, rows, cols) where {T, I <: Integer, V<:AbstractMatrix{T}}
        return new{T, I, V}(blocks, lasts, rows, cols)
    end
end

function AlmostBlockDiagonal(blocks::Vector{V}, lasts::Vector{I}) where {T, I <: Integer, V<:AbstractMatrix{T}}
    rows_and_cols = size.(blocks)
    rows = first.(rows_and_cols)
    cols = last.(rows_and_cols)
    return AlmostBlockDiagonal{T, I, V}(blocks, lasts, rows, cols)
end

AlmostBlockDiagonal(A::AlmostBlockDiagonal) = A

"""
Intermediate matrix used for the representation of modified almost block diagonals mainly
use for pivoting LU factorization.
"""
@concrete struct IntermediateAlmostBlockDiagonal{T, I <: Integer, V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    lasts::Vector{I}
    rows::Vector{I}
    cols::Vector{I}
    fillers::Union{Vector{I}, Nothing}

    function IntermediateAlmostBlockDiagonal{T, I, V}(blocks::Vector{V}, lasts, rows, cols, fillers) where {T, I <: Integer, V <: AbstractMatrix{T}}
        return new{T, I, V}(blocks, lasts, rows, cols, fillers)
    end
end

function IntermediateAlmostBlockDiagonal(blocks::Vector{V}, lasts::Vector{I}, fillers::Vector{I}) where {T, I <: Integer, V<:AbstractMatrix{T}}
    rows_and_cols = size.(blocks)
    rows = first.(rows_and_cols)
    cols = last.(rows_and_cols)
    return IntermediateAlmostBlockDiagonal{T, I, V}(blocks, lasts, rows, cols, fillers)
end

# construct an intermediate almost block diagonals directly from blocks and lasts input
function IntermediateAlmostBlockDiagonal(blocks::Vector{V}, lasts::Vector{I}) where {T, I <: Integer, V<:AbstractMatrix{T}}
    rows_and_cols = size.(blocks)
    rows = first.(rows_and_cols)
    cols = last.(rows_and_cols)
    return IntermediateAlmostBlockDiagonal{T, I, V}(blocks, lasts, rows, cols, nothing)
end

IntermediateAlmostBlockDiagonal(A::IntermediateAlmostBlockDiagonal) = A

"""

Convert a `AlmostBlockDiagonal` matrix to an intermediate form to
do the factorization and solving

! note
    Only for the square almost diagonal matrix of course
"""
function IntermediateAlmostBlockDiagonal(A::AlmostBlockDiagonal)
    accumulate_rows = cumsum(A.rows)
    accumulate_lasts = cumsum(A.lasts)
    offset = accumulate_rows .- accumulate_lasts
    new_blocks = [first(A.blocks)]
    for i in 2:length(A.rows)
        if offset[i-1] !== 0
            tmp_block = vcat(zeros(offset[i-1], A.cols[i]), A.blocks[i])
            push!(new_blocks, tmp_block)
        elseif offset[i-1] == 0
            push!(new_blocks, A.blocks[i])
        end
    end
    return IntermediateAlmostBlockDiagonal(new_blocks, A.lasts, offset)
end

@inline is_square(M::AbstractMatrix) = size(M, 1) == size(M, 2)

blocks(A::AlmostBlockDiagonal) = A.blocks

blocksize(A::AlmostBlockDiagonal, p::Integer) = size(blocks(A)[p])
blocksize(A::AlmostBlockDiagonal, p::Integer, q::Integer)=size(blocks(A)[p], 1), size(blocks(A)[q], 2)

"""
    nblocks(A::AlmostBlockDiagonal[, dim])

Return the number of on-diagonal blocks.
"""
nblocks(A::AlmostBlockDiagonal) = length(blocks(A))

Base.size(A::AlmostBlockDiagonal) = (sum(size(A.blocks[i], 1) for i =1:nblocks(A)), sum(A.lasts))
Base.deepcopy(A::AlmostBlockDiagonal) = AlmostBlockDiagonal(deepcopy(A.blocks), A.lasts)

Base.getindex(A::AlmostBlockDiagonal, i::Integer) = A.blocks[i]
function Base.getindex(A::AlmostBlockDiagonal, i::Integer, j::Integer)
    nth_block_row, offset = check_index(A, i)
    accumulate_lasts = cumsum(A.lasts)
    jₐ = (nth_block_row == 1) ? 0 : accumulate_lasts[nth_block_row-1]
    jᵦ = (nth_block_row == 1) ? A.cols[1] : accumulate_lasts[nth_block_row-1]+A.cols[nth_block_row]
    if jₐ < j <= jᵦ
        ith_block = A.blocks[nth_block_row]
        @inbounds return ith_block[offset, j-jₐ]
    else
        @inbounds return 0.0
    end
end

function Base.Matrix(IA::IntermediateAlmostBlockDiagonal{T}) where T
    N = nblocks(IA)
    W = zeros(T, size(IA))
    rowsᵢ = cumsum(IA.rows) .- IA.rows .+ 1
    rowsᵢ₊₁ = cumsum(IA.rows)
    colsᵢ = cumsum(IA.lasts[1:end-1]) .+ 1
    colsᵢ₊₁ = colsᵢ + IA.cols[2:end] .- 1
    for i in 1:N
        if i == 1
            W[1:IA.rows[1], 1:IA.cols[1]] = IA.blocks[1]
            continue
        end
        @views W[rowsᵢ[i]:rowsᵢ₊₁[i], colsᵢ[i-1]:colsᵢ₊₁[i-1]] = IA.blocks[i]
    end
    return W
end

function Base.Matrix(A::AlmostBlockDiagonal{T}) where T
    N = nblocks(A)
    W = zeros(T, size(A))
    rowsᵢ = cumsum(A.rows) .- A.rows .+ 1
    rowsᵢ₊₁ = cumsum(A.rows)
    colsᵢ = cumsum(A.lasts[1:end-1]) .+ 1
    colsᵢ₊₁ = colsᵢ + A.cols[2:end] .- 1
    for i in 1:N
        if i == 1
            W[1:A.rows[1], 1:A.cols[1]] = A.blocks[1]
            continue
        end
        @views W[rowsᵢ[i]:rowsᵢ₊₁[i], colsᵢ[i-1]:colsᵢ₊₁[i-1]] = A.blocks[i]
    end
    return W
end

function Base.:*(A::AlmostBlockDiagonal{T, I, V}, x::AbstractVector{S}) where {I <: Integer, V <: AbstractArray, T, S}
    MA = Matrix(A) * x
    return MA
end

# ignore varying size of similar for now, since creating a larger size of a given
# ABD matrix without providing additional blocks is nonsense
function Base.similar(A::AlmostBlockDiagonal, ::Type{T}) where {T}
    return AlmostBlockDiagonal(map(x -> similar(x, T), blocks(A)), A.lasts)
end

function Base.zero(A::AlmostBlockDiagonal{T, I, V}) where {V <: AbstractArray, I <: Integer, T}
    AlmostBlockDiagonal(zero.(blocks(A)), A.lasts)
end

function Base.similar(A::IntermediateAlmostBlockDiagonal, ::Type{T}) where {T}
    return IntermediateAlmostBlockDiagonal(map(x -> similar(x, T), blocks(A)), A.lasts)
end

function Base.zero(A::IntermediateAlmostBlockDiagonal{T, I, V}) where {V <: AbstractArray, I <: Integer, T}
    IntermediateAlmostBlockDiagonal(zero.(blocks(A)), A.lasts)
end

# check `i` located in m-th row
function check_index(A::AlmostBlockDiagonal, i::Integer)
    accumulate_rows = cumsum(A.rows)
    for j in eachindex(accumulate_rows)
        if i <= accumulate_rows[j]
            (j == 1) && return 1, i
            return j, i - accumulate_rows[j-1]
        end
    end
end

getblock(A::AlmostBlockDiagonal, p::Integer) = blocks(A)[p]
function getblock(A::AlmostBlockDiagonal{T}, p::Integer, q::Integer) where T
    return p == q ? blocks(A)[p] : zeros{T}(blocksize(B, p, q))
end

blocks(A::IntermediateAlmostBlockDiagonal) = A.blocks

blocksize(A::IntermediateAlmostBlockDiagonal, p::Integer) = size(blocks(A)[p])
function blocksize(A::IntermediateAlmostBlockDiagonal, p::Integer, q::Integer)
    return size(blocks(A)[p], 1), size(blocks(A)[q], 2)
end

"""
nblocks(A::IntermediateAlmostBlockDiagonal[, dim])
"""
nblocks(A::IntermediateAlmostBlockDiagonal) = length(blocks(A))

Base.size(A::IntermediateAlmostBlockDiagonal) = (sum(size(A.blocks[i], 1) for i =1:nblocks(A)), sum(A.lasts))

Base.getindex(A::IntermediateAlmostBlockDiagonal, i::Integer) = A.blocks[i]
function Base.getindex(A::IntermediateAlmostBlockDiagonal, i::Integer, j::Integer)
    nth_block_row, offset = check_index(A, i)
    accumulate_lasts = cumsum(A.lasts)
    jₐ = (nth_block_row == 1) ? 0 : accumulate_lasts[nth_block_row-1]
    jᵦ = (nth_block_row == 1) ? A.cols[1] : accumulate_lasts[nth_block_row-1]+A.cols[nth_block_row]
    if jₐ < j <= jᵦ
        ith_block = A.blocks[nth_block_row]
        @inbounds return ith_block[offset, j-jₐ]
    else
        @inbounds return 0.0
    end
end

function Base.setindex!(A::AlmostBlockDiagonal, v, i, j)
    nth_block_row, offset = check_index(A, i)
    accumulate_lasts = cumsum(A.lasts)
    jₐ = (nth_block_row == 1) ? 0 : accumulate_lasts[nth_block_row-1]
    jᵦ = (nth_block_row == 1) ? A.cols[1] : accumulate_lasts[nth_block_row-1]+A.cols[nth_block_row]
    if jₐ < j <= jᵦ
        A.blocks[nth_block_row][offset, j-jₐ] = v
    else
        throw(ArgumentError(
        "Cannot set entry ($i, $j) in off-almost-diagonal-block to nonzero value $v."
    ))
    end
end

function Base.setindex!(A::IntermediateAlmostBlockDiagonal, v, i, j)
    nth_block_row, offset = check_index(A, i)
    accumulate_lasts = cumsum(A.lasts)
    jₐ = (nth_block_row == 1) ? 0 : accumulate_lasts[nth_block_row-1]
    jᵦ = (nth_block_row == 1) ? A.cols[1] : accumulate_lasts[nth_block_row-1]+A.cols[nth_block_row]
    if jₐ < j <= jᵦ
        A.blocks[nth_block_row][offset, j-jₐ] = v
    else
        throw(ArgumentError(
        "Cannot set entry ($i, $j) in off-almost-diagonal-block to nonzero value $v."
    ))
    end
end

Base.fill!(A::AlmostBlockDiagonal, x) = fill!.(A.blocks, x);
Base.fill!(A::IntermediateAlmostBlockDiagonal, x) = fill!.(A.blocks, x);

Base.:/(A::AlmostBlockDiagonal, n::Number) = AlmostBlockDiagonal(map(x -> x/n, blocks(A)), A.lasts)

# check `i` located in m-th row
function check_index(A::IntermediateAlmostBlockDiagonal, i::Integer)
    accumulate_rows = cumsum(A.rows)
    for j in eachindex(accumulate_rows)
        if i <= accumulate_rows[j]
            (j == 1) && return 1, i
            return j, i - accumulate_rows[j-1]
        end
    end
end

getblock(A::IntermediateAlmostBlockDiagonal, p::Integer) = blocks(A)[p]
function getblock(A::IntermediateAlmostBlockDiagonal{T}, p::Integer, q::Integer) where T
    return p == q ? blocks(A)[p] : zeros{T}(blocksize(B, p, q))
end

function Base.:\(A::AlmostBlockDiagonal{T}, B::AbstractVecOrMat{T2}) where {T, T2 <: Real}
    iflag = 0
    CA = deepcopy(A)
    IA = IntermediateAlmostBlockDiagonal(CA)
    scrtch = zeros(T2, first(size(IA)))
    ipivot = zeros(Integer, first(size(IA)))
    iflag = @views factor_shift(IA, ipivot, scrtch)
    (iflag == 1) && return
    C = deepcopy(B)
    @views substitution(IA, ipivot, C)
    # when A is in the form of ABD with TOPBLK and BOTBLK, the first value in the solution
    # should be the opposite sign
    (A.lasts[1] == 0) ? (C[1] = -C[1]) : nothing
    return C
end

"""
    factor_shift(IA::IntermediateAlmostBlockDiagonal, ipivot::AbstractArray{I}, scrtch)

Factorize the intermediate representation of almost block diagonals matrix `IA[i]` and then shift to prepare the next block `IA[i+1]`.
"""
function factor_shift(IA::IntermediateAlmostBlockDiagonal{T}, ipivot::AbstractArray{I}, scrtch) where {I <: Integer, T}
    info = 0
    indexx = 1
    i = 1
    nbloks = nblocks(IA)

    while true
        nrow = IA.rows[i]
        last = IA.lasts[i]

        @views bloks = IA[i]
        info = @views factor(bloks, ipivot[indexx:indexx+nrow-1], scrtch, last, info)
        (info !== 0) && break

        i == nbloks && return info
        i = i+1
        indexx = indexx + last
        @views bloks_next = IA[i]
        @views shift(bloks, last, bloks_next)
    end
    info = info + indexx - 1
    return info
end

function factor(w::AbstractArray{T}, ipivot::AbstractArray{I}, d, last::I, info::I) where {I <: Integer, T}
    nrow, ncol = size(w)
    d[1:nrow] .= T(0) # don't reassign values

    for j = 1:ncol
        for i = 1:nrow
          d[i] = max(d[i], abs(w[i, j]))
        end
    end
    k = 1
    while k <= last
        (d[k] == 0.0) && (@goto n90)
        (k == nrow) && (@goto n80)
        l = k
        kp1 = k+1
        colmax = abs(w[k, k]) / d[k]

        for i = kp1:nrow
            if abs(w[i, k]) > colmax * d[i] 
                colmax = abs(w[i, k]) / d[i]
                l = i
            end
        end
        ipivot[k] = l
        t = copy(w[l, k])
        s = d[l]
        if l !== k
            w[l, k] = copy(w[k, k])
            w[k, k] = copy(t)
            d[l] = copy(d[k])
            d[k] = copy(s)
        end
        (abs(t)+d[k] <= d[k]) && (@goto n90)

        t = -1.0/t
        w[kp1:nrow, k] .= w[kp1:nrow, k] .* t
        for j=kp1:ncol
            t = copy(w[l, j])
            if l !== k
                w[l,j] = copy(w[k,j])
                w[k,j] = copy(t)
            end
            (t !== 0) && (w[kp1:nrow, j] .+= w[kp1:nrow, k] .* t)
        end
        k = kp1
    end
    return info

    @label n80

    (abs(w[nrow, nrow]) + d[nrow] > d[nrow]) && return info

    @label n90

    info = k
    return info
end

function shift(ai::AbstractArray{T}, last::I, ai1::AbstractArray{T}) where {I <: Integer, T}
    nrowi, ncoli = size(ai)
    ncoli1, _ = size(ai1)
    mmax = nrowi - last
    jmax = ncoli - last
    (mmax < 1 || jmax < 1)   &&   return

    ai1[1:mmax, 1:jmax] = copy(ai[(last+1):(last+mmax), (last+1):(last+jmax)])
    (jmax == ncoli1)  &&   return

    ai1[1:mmax, jmax+1:ncoli1] .= T(0)
    return
end

@views function recursive_unflatten!(y::Vector{Vector{T}}, x::AbstractArray) where {T}
    i = 0
    for yᵢ in y
        copyto!(yᵢ, x[(i + 1):(i + length(yᵢ))])
        i += length(yᵢ)
    end
end

"""
    substitution(IA::IntermediateAlmostBlockDiagonal, ipivot::AbstractArray{I}, x)

Proceed with forward and backward substitution to solve the ADB linear system
"""
function substitution(IA::IntermediateAlmostBlockDiagonal, ipivot::AbstractArray{I}, x::AbstractArray{T}) where {I <: Integer, T <: Real}
    #  forward substitution

    indexx = 1
    last = 0
    nbloks = nblocks(IA)
    for i = 1:nbloks
	    nrow = IA.rows[i]
	    last = IA.lasts[i]

        @views bloks = IA[i]
	    @views forward_substitution(bloks, ipivot[indexx:indexx+last-1], last, x[indexx:indexx+nrow-1])

        indexx = indexx + last
    end

    #  back substitution

    nbp1::Int = nbloks + 1
    for j = 1:nbloks
	    i = nbp1 - j
	    ncol::Int = IA.cols[i]
	    last = IA.lasts[i]
	    indexx = indexx - last

        @views bloks = IA[i]
        @views backward_substitution(bloks, last, x[indexx:indexx+ncol-1])
    end
    return
end
# another dispatch for substitution when input x is a vector of vectors
function substitution(IA::IntermediateAlmostBlockDiagonal, ipivot::AbstractArray{I}, vx::Vector{Vector{T}}) where {I <: Integer, T}
    x = reduce(vcat, vx)
    #  forward substitution

    indexx = 1
    last = 0
    nbloks = nblocks(IA)
    for i = 1:nbloks
	    nrow = IA.rows[i]
	    last = IA.lasts[i]

        @views bloks = IA[i]
	    @views forward_substitution(bloks, ipivot[indexx:indexx+last-1], last, x[indexx:indexx+nrow-1])

        indexx = indexx + last
    end

    #  back substitution

    nbp1::Int = nbloks + 1
    for j = 1:nbloks
	    i = nbp1 - j
	    ncol::Int = IA.cols[i]
	    last = IA.lasts[i]
	    indexx = indexx - last

        @views bloks = IA[i]
        @views backward_substitution(bloks, last, x[indexx:indexx+ncol-1])
    end
    recursive_unflatten!(vx, x)
    return
end

function forward_substitution(w::AbstractArray{T}, ipivot::AbstractArray{I}, last::I, x) where {I <: Integer, T}
    nrow, _ = size(w)
    nrow == 1       &&        return
    lstep = min(nrow-1, last)
    for k = 1:lstep
	    kp1 = k + 1
	    ip = ipivot[k]
	    t = x[ip]
	    x[ip] = x[k]
	    x[k] = t
	    (t !== 0.0) && (x[kp1:nrow] .+= w[kp1:nrow, k] .* t)
    end
    return
end

function backward_substitution(w::AbstractArray{T}, last::I, x) where {I <: Integer, T}
    _, ncol = size(w)
    lp1 = last + 1
    if lp1 <= ncol
        for j = lp1:ncol
	        t = - x[j]
	        (t !== 0.0) && (x[1:last] .+= w[1:last, j] .* t)
        end
    end
    if last !== 1
        lm1 = last - 1
        for kb = 1:lm1
            km1 = last - kb
            k = km1 + 1
            x[k] = x[k]/w[k, k]
            t = - x[k]
            (t !== 0.0) && (x[1:km1] .+= w[1:km1, k] .* t)
        end
    end
    x[1] = x[1]/w[1, 1]
    return
end

export AlmostBlockDiagonal, IntermediateAlmostBlockDiagonal
export factor_shift, substitution

end
