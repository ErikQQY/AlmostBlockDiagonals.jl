module AlmostBlockDiagonals

using ConcreteStructs
import Base.\

"""
    AlmostBlockDiagonal(T, V<:AbstractMatrix{T}) < AbstractMatrix{T}

A matrix with matrices on the diagonal, but not strictly has their corner against each other.

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

! note
    The column of block `ncol` and row of block `nrow` must satisfy: `ncol` ≥ `nrow`.
"""
@concrete struct AlmostBlockDiagonal{T, I, V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    lasts::Vector{I}
    rows::Vector{I}
    cols::Vector{I}

    function AlmostBlockDiagonal{T, I, V}(blocks::Vector{V}, lasts, rows, cols) where {T, I, V<:AbstractMatrix{T}}
        return new{T, I, V}(blocks, lasts, rows, cols)
    end
end

function AlmostBlockDiagonal(blocks::Vector{V}, lasts::Vector{I}) where {T, I, V<:AbstractMatrix{T}}
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
@concrete struct IntermediateAlmostBlockDiagonal{T, I, V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    lasts::Vector{I}
    rows::Vector{I}
    cols::Vector{I}
    fillers::Vector{I}

    function IntermediateAlmostBlockDiagonal{T, I, V}(blocks::Vector{V}, lasts, rows, cols, fillers) where {T, I, V <: AbstractMatrix{T}}
        return new{T, I, V}(blocks, lasts, rows, cols, fillers)
        
    end
end

function IntermediateAlmostBlockDiagonal(blocks::Vector{V}, lasts::Vector{I}, fillers::Vector{I}) where {T, I, V<:AbstractMatrix{T}}
    rows_and_cols = size.(blocks)
    rows = first.(rows_and_cols)
    cols = last.(rows_and_cols)
    return IntermediateAlmostBlockDiagonal{T, I, V}(blocks, lasts, rows, cols, fillers)
end

IntermediateAlmostBlockDiagonal(A::IntermediateAlmostBlockDiagonal) = A

"""

Convert a `AlmostBlockDiagonal` matrix to an intermediate form to
do factorization

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
function blocksize(A::AlmostBlockDiagonal, p::Integer, q::Integer)
    return size(blocks(A)[p], 1), size(blocks(A)[q], 2)
end


"""
    nblocks(A::AlmostBlockDiagonal[, dim])

Return the number of on-diagonal blocks.
"""
nblocks(A::AlmostBlockDiagonal) = length(blocks(A))

Base.size(A::AlmostBlockDiagonal) = (sum(size(A.blocks[i], 1) for i =1:nblocks(A)), sum(A.lasts))

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
function getblock(A::AlmostBlockDiagonal, p::Integer, q::Integer) where T
    return p == q ? blocks(A)[p] : Zeros{T}(blocksize(B, p, q))
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
function getblock(A::IntermediateAlmostBlockDiagonal, p::Integer, q::Integer) where T
    return p == q ? blocks(A)[p] : Zeros{T}(blocksize(B, p, q))
end

function Base.:\(A::AlmostBlockDiagonal{T}, B::AbstractVecOrMat{T2}) where {T, T2}
    iflag = 1
    IA = IntermediateAlmostBlockDiagonal(A)
    x = zeros(T, size(A)[1])
    B = preprocess_B(A, IA, B)
    ipivot = zeros(Integer, length(B))
    fcblok(IA, ipivot, x, iflag)
    (iflag == 0) && return
    substitution(IA, ipivot, B, x)
    return x
end

function preprocess_B(A::AlmostBlockDiagonal, IA::IntermediateAlmostBlockDiagonal, B::AbstractArray)
    offset = IA.fillers
    accumulate_rows = cumsum(A.rows)
    newB = B[1:accumulate_rows[1]]
    for i in 2:length(IA.rows)
        if offset[i-1] !== 0
            tmpB = vcat(zeros(offset[i-1]), B[(accumulate_rows[i-1]+1):accumulate_rows[i]])
            newB = vcat(newB, tmpB)
        else
            newB = vcat(newB, B[(accumulate_rows[i-1]+1):accumulate_rows[i]])
        end
    end
    return newB
end

function fcblok(IA, ipivot, scrtch, iflag)
    iflag = 1
    indexb = 1
    i = 1
    nbloks = nblocks(IA)
    while true    
        nrow = IA.rows[i]
        ncol = IA.cols[i]
        last = IA.lasts[i]

        @views blok = IA[i]
        @views factor(blok, ipivot[indexb:(indexb+nrow-1)], scrtch, nrow, ncol, last, iflag)

        ((iflag == 0) || (i == nbloks)) && return
        i = i+1
        ncoli = IA.cols[i]

        @views blok_next = IA[i]
        @views shift(blok,ipivot[indexb:(indexb+nrow-1)],nrow,ncol,last,blok_next, ncoli)
        indexb = indexb + nrow
    end
end
function factor(w, ipivot, d, nrow, ncol, last, iflag)
    for i = 1:nrow
        ipivot[i] = i
        rowmax = 0.0
        for j=1:ncol
            rowmax = max(rowmax, abs(w[i,j]))
        end
        if rowmax == 0
            iflag = 0
            return
        end
        d[i] = rowmax
    end

    k = 1
    ipivk = 0
    while k <= last
        ipivk = ipivot[k]
        if k == nrow
            if abs(w[ipivk, nrow]) + d[ipivk] > d[ipivk]
                return
            end
        end
        j = k
        kp1 = k+1
        colmax = abs(w[ipivk, k])/d[ipivk]
        for i=kp1:nrow
            ipivi = ipivot[i]
            awikdi = abs(w[ipivi, k])/d[ipivi]
            if awikdi <= colmax
                continue
            end
            colmax = awikdi
            j = i
        end
        if j !== k
            ipivk = ipivot[j]
            ipivot[j] = ipivot[k]
            ipivot[k] = ipivk
            iflag = -iflag
        end

        if abs(w[ipivk, k]) + d[ipivk] <= d[ipivk]
            iflag = 0
            return
        end
        for i = kp1:nrow
            ipivi = ipivot[i]
            w[ipivi, k] = w[ipivi, k]/w[ipivk, k]
            ratio = -w[ipivi, k]
            for j = kp1:ncol
                w[ipivi, j] = ratio*w[ipivk, j] + w[ipivi, j]
            end
        end
        k = kp1
    end
    if abs(w[ipivk, k]) + d[ipivk] <= d[ipivk]
        iflag = 0
        return
    end
end

function shift(ai, ipivot, nrowi, ncoli, last, ai1, ncoli1)
    mmax = nrowi - last
    jmax = ncoli - last
    ((mmax < 1) || (jmax < 1)) && return

    for m=1:mmax
        ip = ipivot[last+m]
        for j=1:jmax
            ai1[m,j] = ai[ip,last+j]
        end
    end
    (jmax == ncoli1) && return

    jmaxp1 = jmax + 1
    for j=jmaxp1:ncoli1
        for m=1:mmax
            ai1[m,j] = 0.0
        end
    end
end

function substitution(IA, ipivot, b, x)
    index = 1
    indexb = 1
    indexx = 1
    nbloks = nblocks(IA)
    for i=1:nbloks
        nrow = IA.rows[i]
        ncol = IA.cols[i]
        last = IA.lasts[i]
        @views blok = IA[i]
        @views forward_substitution(blok,ipivot[indexb:(indexb+nrow-1)],nrow,last,b[indexb:(indexb+2*nrow-last-1)], x[indexx:(indexx+nrow-1)])
        index = nrow*ncol + index
        indexb = indexb + nrow
        indexx = indexx + last
    end

    nbp1 = nbloks + 1
    for j=1:nbloks
         i = nbp1 - j
         nrow = IA.rows[i]
         ncol = IA.cols[i]
         last = IA.lasts[i]
         index = index - nrow*ncol
         indexb = indexb - nrow
         indexx = indexx - last
         @views blok = IA[i]
        @views backward_substitution(blok,ipivot[indexb:(indexb+nrow-1)],ncol,last,x[indexx:(indexx+ncol-1)])
    end
end
function forward_substitution(w, ipivot, nrow, last, b, x)
    ip = ipivot[1]
    x[1] = b[ip]
    (nrow == 1) && return
    for k=2:nrow
        ip = ipivot[k]
        jmax = min(k-1,last)
        sum = 0.0
        for j=1:jmax
            sum = w[ip,j]*x[j] + sum
        end
        x[k] = b[ip] - sum
    end
    nrowml = nrow - last
    (nrowml == 0) && return
    lastp1 = last+1
    for k=lastp1:nrow
        b[nrowml+k] = x[k]
    end
end
function backward_substitution(w, ipivot, ncol, last, x)
    k = last
    ip = ipivot[k]
    sum = 0.0
    if k !== ncol
        kp1 = k+1
        @label backward_substitution2
        for j=kp1:ncol
            sum = w[ip,j]*x[j] + sum
        end
    end
    x[k] = (x[k] - sum)/w[ip,k]
    (k == 1) && return
    kp1 = k
    k = k-1
    ip = ipivot[k]
    sum = 0.0
    @goto backward_substitution2
end

export AlmostBlockDiagonal, IntermediateAlmostBlockDiagonal
export slvblk

end


#=


iflag = 1
x = zeros(Float64, 11)
slvblk(A, OB, ipivot, x, iflag)
=#