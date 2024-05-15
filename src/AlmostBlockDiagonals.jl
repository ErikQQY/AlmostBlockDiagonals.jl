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
    fillers::Vector{I}

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

function Base.:\(A::AlmostBlockDiagonal{T}, B::AbstractVecOrMat{T2}) where {T, T2}
    iflag = 1
    CA = deepcopy(A)
    IA = IntermediateAlmostBlockDiagonal(CA)
    scrtch = zeros(T2, last(size(A)))
    ipivot = zeros(Integer, last(size(A)))
    @views factor_shift(IA, ipivot, scrtch)
    (iflag == 0) && return
    @views substitution(IA, ipivot, B)
    return B
end

function factor_shift(IA::IntermediateAlmostBlockDiagonal{T}, ipivot::AbstractArray{I}, scrtch) where {I <: Integer, T}
    info = 0
    indexx = 1
    i = 1
    nbloks = nblocks(IA)

    while true
        nrow = IA.rows[i]
        ncol = IA.cols[i]
        last = IA.lasts[i]

        @views bloks = IA[i]
        info = @views factor(bloks, ipivot[indexx:indexx+nrow-1], scrtch, nrow, ncol, last, info)
        (info !== 0) && break

        i == nbloks && return info
        i = i+1
        indexx = indexx + last
        ncol_next = IA.cols[i]
        @views bloks_next = IA[i]
        @views shift(bloks, nrow, ncol, last, bloks_next, ncol_next)
    end
    info = info + indexx - 1
    return info
end

function factor(w::AbstractArray{T}, ipivot::AbstractArray{I}, d, nrow::I, ncol::I, last::I, info::I) where {I <: Integer, T}
    d = zeros(T, nrow)

    for j = 1:ncol
        for i = 1:nrow
          d[i] = max(d[i], abs(w[i, j]))
        end
    end
    k = 1
    while k <= last
        if d[k] == 0.0
            return k
        end
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
        t = w[l, k]
        s = d[l]
        if l !== k
            w[l, k] = w[k, k]
            w[k, k] = t
            d[l] = d[k]
            d[k] = s
        end
        if abs(t)+d[k] <= d[k]
            return k
        end

        t = -1.0/t
        for i = kp1:nrow
            w[i, k] = w[i, k] * t
        end
        for j=kp1:ncol
            t = w[l, j]
            if l !== k
                w[l,j] = w[k,j]
                w[k,j] = t
            end
            if t !== 0
                for i = kp1:nrow
                    w[i,j] = w[i,j] + w[i,k] * t
                end
            end
        end
        k = kp1
    end
    return info

    @label n80

    (abs(w[nrow, nrow])+d[nrow] > d[nrow]) && return info
    info = k
    return info
end

function shift(ai::AbstractArray{T}, nrowi::I, ncoli::I, last::I, ai1::AbstractArray{T}, ncoli1::I) where {I <: Integer, T}
    mmax = nrowi - last
    jmax = ncoli - last
    (mmax < 1 || jmax < 1)   &&   return

    for j=1:jmax
        for m=1:mmax
            ai1[m, j] = ai[last+m,last+j]
        end
    end
    (jmax == ncoli1)  &&   return

    jmaxp1 = jmax + 1
    for j=jmaxp1:ncoli1
	    for m=1:mmax
            ai1[m, j] = 0.0
        end
    end
    return
end

function substitution(IA::IntermediateAlmostBlockDiagonal, ipivot::AbstractArray{I}, x) where {I <: Integer}

#  forward substitution

    indexx = 1
    last = 0
    nbloks = nblocks(IA)
    for i = 1:nbloks
	    nrow = IA.rows[i]
	    last = IA.lasts[i]

        @views bloks = IA[i]
	    @views forward_substitution(bloks, ipivot[indexx:indexx+last-1], nrow, last, x[indexx:indexx+nrow-1])

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
        @views backward_substitution(bloks, ncol, last, x[indexx:indexx+ncol-1])
    end
    return
end

function forward_substitution(w::AbstractArray{T}, ipivot::AbstractArray{I}, nrow::I, last::I, x) where {I <: Integer, T}
    nrow == 1       &&        return
    lstep = min(nrow-1 , last)
    for k = 1:lstep
	    kp1 = k + 1
	    ip = ipivot[k]
	    t = x[ip]
	    x[ip] = x[k]
	    x[k] = t
	    if t !== 0.0
            for i = kp1:nrow
                x[i] = x[i] + w[i,k] * t
            end
        end
    end
    return
end

function backward_substitution(w::AbstractArray{T}, ncol::I, last::I, x) where {I <: Integer, T}
    lp1 = last + 1
    if ( lp1 <= ncol )
        for j = lp1:ncol
	        t = - x[j]
	        if t !== 0.0
	            for i = 1:last
                    x[i] = x[i] + w[i, j] * t
                end
            end
        end
    end
    if last !== 1
        lm1 = last - 1
        for kb = 1:lm1
            km1 = last - kb
            k = km1 + 1
            x[k] = x[k]/w[k, k]
            t = - x[k]
            if t !== 0.0
                for i = 1:km1
                    x[i] = x[i] + w[i, k] * t
                end
            end
        end
    end
    x[1] = x[1]/w[1, 1]
    return
end


## to deprecate

function fcblok(bloks, integs, nbloks, ipivot, scrtch)
    info = 0
    indexx = 1
    indexn = 1
    i = 1

    while true
        index = copy(indexn)
        nrow::Int = integs[1,i]
        ncol::Int = integs[2,i]
        last::Int = integs[3,i]
        info = @views factrb(bloks[index:index+nrow*ncol-1], ipivot[indexx:indexx+nrow-1], scrtch, nrow, ncol, last, info)
        (info !== 0) && break

        i == nbloks && return info
        i = i+1
        indexn::Int = nrow * ncol + index
        indexx::Int = indexx + last

        @views shiftb(bloks[index:index+nrow*ncol-1], nrow, ncol, last, bloks[indexn:indexn+Int(integs[1,i])*Int(integs[2,i])-1], Int(integs[1,i]), Int(integs[2,i]))
    end
    info = info + indexx - 1
    return info
end

function factrb(w, ipivot, d, nrow, ncol, last, info)
    w = reshape(w, nrow, ncol)
    # initialize  d
    for  i = 1:nrow
        d[i] = 0.0
    end

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
        ( abs(t)+d[k] <= d[k] )  &&  (@goto n90)

        t = -1.0/t
        for i = kp1:nrow
            w[i, k] = w[i, k] * t
        end
        for j=kp1:ncol
            t = copy(w[l, j])
            if l !== k
                w[l,j] = copy(w[k,j])
                w[k,j] = copy(t)
            end
            if ( t !== 0 )
                for i = kp1:nrow
                    w[i,j] = w[i,j] + w[i,k] * t
                end
            end
        end
        k = kp1
    end
    return info

    @label n80

    (abs(w[nrow, nrow])+d[nrow] > d[nrow]) && return info

    @label n90 
    info = k
    return info
end

function shiftb(ai, nrowi::Int, ncoli::Int, last, ai1, nrowi1::Int, ncoli1::Int)
    ai = reshape(ai, nrowi, ncoli)
    ai1 = reshape(ai1, nrowi1, ncoli1)
    mmax = nrowi - last
    jmax = ncoli - last
    (mmax < 1 || jmax < 1)   &&   return

#  put the remainder of block i into ai1

    for j=1:jmax
        for m=1:mmax
            ai1[m, j] = ai[last+m,last+j]
        end
    end
    (jmax == ncoli1)  &&   return

#  zero out the upper right corner of ai1

    jmaxp1 = jmax + 1
    for j=jmaxp1:ncoli1
	    for m=1:mmax
            ai1[m, j] = 0.0
        end
    end

    return
end

function sbblok(bloks, integs, nbloks, ipivot, x)
#  forward substitution pass

    index::Int = 1
    indexx::Int = 1
    last::Int = 0
    for i = 1:nbloks
	    nrow::Int = integs[1,i]
        ncol::Int = integs[2,i]
	    last = integs[3,i]
	    @views subfor(bloks[index:index+nrow*ncol-1], ipivot[indexx:indexx+last-1], nrow, ncol, last, x[indexx:indexx+nrow-1])

        index = nrow * integs[2,i] + index
        indexx = indexx + last
    end
    
#  back substitution pass

    nbp1::Int = nbloks + 1
    for j = 1:nbloks
	    i = nbp1 - j
	    nrow::Int = integs[1,i]
	    ncol::Int = integs[2,i]
	    last = integs[3,i]
	    index = index - nrow * ncol
	    indexx = indexx - last

        @views subbak(bloks[index:index+nrow*ncol-1], nrow, ncol, last, x[indexx:indexx+ncol-1])
    end
    return
end

function subfor(w, ipivot, nrow, ncol, last, x)
    w = reshape(w, nrow, ncol)
    nrow == 1       &&        return
    lstep = min(nrow-1 , last)
    for k = 1:lstep
	    kp1 = k + 1
	    ip::Int = ipivot[k]
	    t = x[ip]
	    x[ip] = x[k]
	    x[k] = t
	    if t !== 0.0
            for i = kp1:nrow
                x[i] = x[i] + w[i,k] * t
            end
        end
    end
    return
end

function subbak(w, nrow, ncol, last, x)
    w = reshape(w, nrow, ncol)
    lp1 = last + 1
    if ( lp1 <= ncol )
        for j = lp1:ncol
	        t = - x[j]
	        if t !== 0.0
	            for i = 1:last
                    x[i] = x[i] + w[i, j] * t
                end
            end
        end
    end
    if last !== 1
        lm1 = last - 1
        for kb = 1:lm1
            km1 = last - kb
            k = km1 + 1
            x[k] = x[k]/w[k, k]
            t = - x[k]
            if t !== 0.0
                for i = 1:km1
                    x[i] = x[i] + w[i, k] * t
                end
            end
        end
    end
    x[1] = x[1]/w[1, 1]
    return
end

export AlmostBlockDiagonal, IntermediateAlmostBlockDiagonal

end
