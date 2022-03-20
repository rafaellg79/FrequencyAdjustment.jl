
# =================================
# Detail
# =================================

"""
    generate_DFT_to_realDFT_indices(L::Int)

Precompute indices for using in the `cis_DFT_to_cos_realDFT!()` function.
"""
function generate_DFT_to_realDFT_indices(L::Int, N_::Val{N}) where N
    # Compute linear indexing of a N-dimensional window with size L
    I = collect(LinearIndices(ntuple(n -> L, N)))
    # Reverse the indices
    reverse!(I, dims=:)
    # Move the center of the reversed window back to the center of the original window
    I = circshift(I, ones(N))
    # Remove the indices of the complex conjugate elements from the window
    rfftM = L÷2 + 1
    I[1:rfftM,ntuple(n -> :, N-1)...] :: Array{Int, N}
end


"""
    Q, L, C = quadratic_lsqfit_log_abs(F::AbstractMatrix{Complex}) :: Tuple

Fit a quadratic surface to the logarithm of the absolute values of the 3x3
complex matrix `F`, returning the coefficients for the quadratic ``q(x,y) = a
x^2 + b x y + c y^2 + d x + e y + f``. No bounds check is performed on `F` and
thus `F[1:9]` should exist.
"""
@inline function quadratic_lsqfit_log_abs(
  F :: AbstractArray{Complex{T}, 2}
) where T <: AbstractFloat
    # If any element of F is 0 then no surface fits log.(abs.(F)) as log(0) == -Inf
    if any(f≈0 for f in F)
        # So return a 0 matrix
        return Symmetric(T.([0 0; 0 0])), T.([0,0]), T(0)
    end
    # Compute the logarithm of the absolute values of the region F to fit the surface
    @inbounds begin
    b1::T = F[1] |> abs |> log
    b2::T = F[2] |> abs |> log
    b3::T = F[3] |> abs |> log
    b4::T = F[4] |> abs |> log
    b5::T = F[5] |> abs |> log
    b6::T = F[6] |> abs |> log
    b7::T = F[7] |> abs |> log
    b8::T = F[8] |> abs |> log
    b9::T = F[9] |> abs |> log
    end

    # Constants
    inv4::T = inv(T(4))
    inv6::T = inv(T(6))
    inv9::T = inv(T(9))

    # Hard coded solution of system
    a::T = (b1 + b2 + b3 - 2*b4 - 2*b5 - 2*b6 + b7 + b8 + b9)*inv6
    b::T = (b1 - b3 - b7 + b9)*inv4
    c::T = (b1 - 2*b2 + b3 + b4 - 2*b5 + b6 + b7 - 2*b8 + b9)*inv6
    d::T = (-8*b1 - 5*b2 - 2*b3 + 8*b4 + 8*b5 + 8*b6 - 3*b8 - 6*b9)*inv6
    e::T = (-8*b1 + 8*b2 - 5*b4 + 8*b5 - 3*b6 - 2*b7 + 8*b8 - 6*b9)*inv6
    f::T = (26*b1 - b2 + 2*b3 - b4 - 19*b5 - 7*b6 + 2*b7 - 7*b8 + 14*b9)*inv9
    
    # Store solution into matrix form
    Q = Symmetric([a b/2; b/2 c])
    L = [d, e]
    C = f

    return Q,L,C
end

"""
    Q,L,C = quadratic_lsqfit_log_abs(F::AbstractArray{Complex,3}) :: Tuple

Fit a quadratic surface to the logarithm of the absolute values of the 3x3x3
complex matrix `F`, returning the coefficients for the quadratic ``q(x,y,z) =
X'Q*X + L*X + C`` where `Q` is a symmetric 3x3 matrix, L is a linear functional,
`C` is a real constant, and `X = [x,y,z]`.  No bounds check is performed on `F` and
thus `F[1:27]` should exist.
"""
@inline function quadratic_lsqfit_log_abs(
  F :: AbstractArray{Complex{T}, 3}
) :: Tuple{ Symmetric{T}, Vector{T}, T } where {T <: AbstractFloat}

    # If any element of F is 0 then no surface fits log.(abs.(F)) as log(0) == -Inf
    if any(f≈0 for f in F)
        # So return a 0 matrix
        return Symmetric(T.([0 0 0; 0 0 0; 0 0 0])), T.([0,0,0]), T(0)
    end
    
    # Constants
    inv18::T = inv(T(18))
    inv24::T = inv(T(24))
    inv27::T = inv(T(27))

    # Compute the logarithm of the absolute values of the region F to fit the surface
    @inbounds begin
    b₀::T = F[1] |> abs |> log
    b₁::T = F[2] |> abs |> log
    b₂::T = F[3] |> abs |> log
    b₃::T = F[4] |> abs |> log
    b₄::T = F[5] |> abs |> log
    b₅::T = F[6] |> abs |> log
    b₆::T = F[7] |> abs |> log
    b₇::T = F[8] |> abs |> log
    b₈::T = F[9] |> abs |> log
    b₉::T = F[10] |> abs |> log
    b₁₀::T = F[11] |> abs |> log
    b₁₁::T = F[12] |> abs |> log
    b₁₂::T = F[13] |> abs |> log
    b₁₃::T = F[14] |> abs |> log
    b₁₄::T = F[15] |> abs |> log
    b₁₅::T = F[16] |> abs |> log
    b₁₆::T = F[17] |> abs |> log
    b₁₇::T = F[18] |> abs |> log
    b₁₈::T = F[19] |> abs |> log
    b₁₉::T = F[20] |> abs |> log
    b₂₀::T = F[21] |> abs |> log
    b₂₁::T = F[22] |> abs |> log
    b₂₂::T = F[23] |> abs |> log
    b₂₃::T = F[24] |> abs |> log
    b₂₄::T = F[25] |> abs |> log
    b₂₅::T = F[26] |> abs |> log
    b₂₆::T = F[27] |> abs |> log
    end

    # Hard coded solution of system
    a::T = (b₀ + b₁ + b₁₀ + b₁₁ - 2.0b₁₂ - 2.0b₁₃ - 2.0b₁₄ + b₁₅ + b₁₆ + b₁₇ + b₁₈ + b₁₉ + b₂ + b₂₀ - 2.0b₂₁ - 2.0b₂₂ - 2.0b₂₃ + b₂₄ + b₂₅ + b₂₆ - 2.0b₃ - 2.0b₄ - 2.0b₅ + b₆ + b₇ + b₈ + b₉)*inv18
    b::T = (b₀ - b₁₁ - b₁₅ + b₁₇ + b₁₈ - b₂ - b₂₀ - b₂₄ + b₂₆ - b₆ + b₈ + b₉)*inv24
    c::T = (b₀ + b₁ - b₁₈ - b₁₉ + b₂ - b₂₀ + b₂₄ + b₂₅ + b₂₆ - b₆ - b₇ - b₈)*inv24
    d::T = (b₀ - 2.0b₁ - 2.0b₁₀ + b₁₁ + b₁₂ - 2.0b₁₃ + b₁₄ + b₁₅ - 2.0b₁₆ + b₁₇ + b₁₈ - 2.0b₁₉ + b₂ + b₂₀ + b₂₁ - 2.0b₂₂ + b₂₃ + b₂₄ - 2.0b₂₅ + b₂₆ + b₃ - 2.0b₄ + b₅ + b₆ - 2.0b₇ + b₈ + b₉)*inv18
    e::T = (b₀ - b₁₈ - b₂ + b₂₀ - b₂₁ + b₂₃ - b₂₄ + b₂₆ + b₃ - b₅ + b₆ - b₈)*inv24
    f::T = (b₀ + b₁ - 2.0b₁₀ - 2.0b₁₁ - 2.0b₁₂ - 2.0b₁₃ - 2.0b₁₄ - 2.0b₁₅ - 2.0b₁₆ - 2.0b₁₇ + b₁₈ + b₁₉ + b₂ + b₂₀ + b₂₁ + b₂₂ + b₂₃ + b₂₄ + b₂₅ + b₂₆ + b₃ + b₄ + b₅ + b₆ + b₇ + b₈ - 2.0b₉)*inv18
    g::T = (-11.0b₀ - 8.0b₁ - 5.0b₁₀ - 2.0b₁₁ + 8.0b₁₂ + 8.0b₁₃ + 8.0b₁₄ - 3.0b₁₆ - 6.0b₁₇ - 5.0b₁₈ - 2.0b₁₉ - 5.0b₂ + b₂₀ + 8.0b₂₁ + 8.0b₂₂ + 8.0b₂₃ - 3.0b₂₄ - 6.0b₂₅ - 9.0b₂₆ + 8.0b₃ + 8.0b₄ + 8.0b₅ + 3.0b₆ - 3.0b₈ - 8.0b₉) * inv18
    h::T = (-11.0b₀ + 8.0b₁ + 8.0b₁₀ - 5.0b₁₂ + 8.0b₁₃ - 3.0b₁₄ - 2.0b₁₅ + 8.0b₁₆ - 6.0b₁₇ - 5.0b₁₈ + 8.0b₁₉ + 3.0b₂ - 3.0b₂₀ - 2.0b₂₁ + 8.0b₂₂ - 6.0b₂₃ + b₂₄ + 8.0b₂₅ - 9.0b₂₆ - 8.0b₃ + 8.0b₄ - 5.0b₆ + 8.0b₇ - 3.0b₈ - 8.0b₉) * inv18
    i::T = (-11.0b₀ - 8.0b₁ + 8.0b₁₀ + 8.0b₁₁ + 8.0b₁₂ + 8.0b₁₃ + 8.0b₁₄ + 8.0b₁₅ + 8.0b₁₆ + 8.0b₁₇ + 3.0b₁₈ - 5.0b₂ - 3.0b₂₀ - 3.0b₂₂ - 6.0b₂₃ - 3.0b₂₄ - 6.0b₂₅ - 9.0b₂₆ - 8.0b₃ - 5.0b₄ - 2.0b₅ - 5.0b₆ - 2.0b₇ + b₈ + 8.0b₉) * inv18
    j::T = (52.0b₀ + 16.0b₁ - 11.0b₁₀ - 8.0b₁₁ - 11.0b₁₂ - 29.0b₁₃ - 17.0b₁₄ - 8.0b₁₅ - 17.0b₁₆ + 4.0b₁₇ + 10.0b₁₈ - 8.0b₁₉ + 10.0b₂ + 4.0b₂₀ - 8.0b₂₁ - 17.0b₂₂ + 4.0b₂₃ + 4.0b₂₄ + 4.0b₂₅ + 34.0b₂₆ + 16.0b₃ - 11.0b₄ - 8.0b₅ + 10.0b₆ - 8.0b₇ + 4.0b₈ + 16.0b₉) * inv27

    # Store solution into matrix form
    Q = Symmetric([a b c; b d e; c e f])
    L = [g,h,i]
    C = j

    return Q,L,C
end

"""
    Q,L,C = quadratic_lsqfit_log_abs(F::AbstractArray{Complex,N}) :: Tuple

Fit a quadratic surface to the logarithm of the absolute values of the 3x3...x3
complex matrix `F`, returning the coefficients for the quadratic ``q(X) =
X'Q*X + L*X + C`` where `Q` is a symmetric NxN matrix, `L` is a linear functional,
`C` is a real constant, and `X = [x1,x2,...,xN]`.  No size check is performed on `F` and
thus `length(F)` must be `3^N`.
"""
@inline function quadratic_lsqfit_log_abs(
  F :: AbstractArray{Complex{T}, N}
) :: Tuple{ Symmetric{T}, Vector{T}, T } where {T <: AbstractFloat} where N
    # If any element of F is 0 then no surface fits log.(abs.(F)) as log(0) == -Inf
    if any(f≈0 for f in F)
        # So return a 0 matrix
        return Symmetric(zeros(T, N, N)), zeros(T, N), T(0)
    end
    
    # Compute Cartesian Indices of F
    X = CartesianIndices(F)
    # Pre-allocate memory to store the linear system q.(X) = B, where q is the quadratic equation q(X)=X'Q*X+L*X+C
    # one equation for each point of the surface F, i.e. length(F), and
    # the N^2 + N + 1 coefficients of the quadratic surface to be fit
    # given the surface q(X)=X'*Q*X+L*X+C we have length(Q)+length(L)+1 variables (note that C is a scalar) or
    # by observing that Q has length(X') rows and length(X) columns and L has length(X) elements then
    # we have N^2+N+1 variables
    A = zeros(length(F), N^2+N+1)
    # Compute the logarithm of the absolute values of the region F to fit the surface
    B = F[1:end] .|> abs .|> log
    
    # For every equation of A
    for i in 1:length(F)
        # Let v be a Vector{Int} with the i-th equation's cartesian coordinate
        v = [X[i].I...]
        # For each pair of coordinates
        for x = 1:N
            for y = 1:N
                # Compute the linear index j of that coordinate
                j = y+(x-1)*N
                # And store in A the values of v'*Q*v that multiplies Q[y, x], that is (v'*ones(size(Q))*v)[y, x]
                # which is equivalent to v'[y]*v[x]
                A[i, j] = v[x] * v[y]
            end
        end
        # Store into A the values of L*v that multiplies L[j], that is v[j]
        for j in 1:N
            A[i,N^2+j] = v[j]
        end
        # Remember that q(X)=X'*Q*X+L*X+C, thus 1 multiplies C and so store 1 into the system
        A[i, end] = 1
    end
    
    # Solve the system via least squares
    QLC = A \ B
    # Create views for the solution
    Q = reshape((@view QLC[N*N:-1:1]), N, N)
    L = @view QLC[N^2+N:-1:N^2+1]
    C = QLC[end]
    
    return Symmetric(Q), L, C
end

"""
    x,y,max = quadratic_critical_point( (a,b,c,d,e,f) ) :: Tuple

Return the location of the critical point of the quadratic surface ``q(x,y) = a
x^2 + b x y + c y^2 + d x + e y + f``, together with the value of ``q`` at such
a point. (A critical point is one where both partial derivatives of the
quadratic vanish).
"""
@inline function quadratic_critical_point(
    Q::Symmetric{T}, L::NTuple{2, T}, C::T
) :: Tuple{NTuple{2, T},T} where T <: AbstractFloat
    # Store coefficients from Q, L and C into local variables
    a::T,b::T,c::T = Q[[1,2,4]]
    b *= 2
    d::T,e::T = L
    f::T = C

    # Hard-coded solution
    invΔ::T = inv(b^2 - 4*a*c)
    x::T = (2*c*d - b*e) * invΔ
    y::T = (-(b*d) + 2*a*e) * invΔ
    max::T = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
    return ((y,x),max) :: Tuple{NTuple{2, T},T}
end

"""
    X,max = quadratic_critical_point( Q,L,C ) :: Tuple

Return the location of the critical point of the quadratic ``q(X) = X'QX + LX + C``
together with the value of ``q`` at such a point. (A critical point
is one where both partial derivatives of the quadratic vanish).
"""
@inline function quadratic_critical_point(
    Q::Symmetric{T}, L::NTuple{N, T}, C::T
) :: Tuple{NTuple{N, T},T} where {T<:AbstractFloat} where N
    # Conversion from tuple to array
    L = collect(L)
    # Solve system ∂q(X)/∂X = 0
    I = 2Q \ -L
    # The solution is the critical point of the surface and thus a local maximum/minimum
    max = dot(I, Q*I) + dot(L,I) + C
    return (tuple(I...),max) :: Tuple{NTuple{N, T},T}
end

@inline function quadratic_critical_point(
    Q::Symmetric{T}, L::Vector{T}, C::T
) where T
    # Fallback to appropriate quadratic_critical_point function based on N from NTuple{N, T} == typeof(tuple(L...))
    quadratic_critical_point(Q, tuple(L...), C)
end

"""
    wraparound(val, max)

Perform a circular wrapping of `val` to the range `[1,max]`.
"""
@inline function wraparound(val, max) mod(val-1, max) + 1 end

ind2sub(sz, ind) = Tuple( CartesianIndices(sz)[ind] )
sub2ind(sz, inds...) = LinearIndices(sz)[inds...]

"""
    realDFT_3x3_view(S::FrequencyAdjuster, F::AbstractMatrix{Complex}, y::Int, x::Int) :: SubArray

Compute a 3x3 view (SubArray) of the real-DFT `F` considering periodic
(wrap-around) boundary conditions (ie, `F = rfft(s)` for some `s`).
"""
@inline function realDFT_3x3_view(
  S :: FrequencyAdjuster{T, 2}
, F :: AbstractArray{Complex{T}, 2}
, y :: Int
, x :: Int
) :: SubArray{Complex{T} ,2} where T <: AbstractFloat
    # Compute the (y, x) index of the real-DFT F considering periodic
    # (wrap-around) boundary conditions.
    @inline function compute_index(y::Int,x::Int) :: Tuple{Int,Int}
        x = wraparound(x,S.L)

        if y <= 0
            y = 2-y
            x = x>1 ? S.L - x + 2 : x
        elseif y > S.rfftM
            y = S.rfftM
            x = x>1 ? S.L - x + 2 : x
        end

        return (y, x) :: Tuple{Int,Int}
    end

    # Create LinearIndices object of the window
    LI = LinearIndices(size(F))
    # Store into inds the linear indices of the 3x3 region around (y, x) after
    # considering periodic boundary conditions.
    inds = [ LI[compute_index(y,x)...] for y in y-1:y+1, x in x-1:x+1 ] :: Matrix{Int}

    # Return a View of F[inds]
    @inbounds return (@view F[inds]) :: SubArray{Complex{T},2}
end

"""
    realDFT_3x3_view(S::FrequencyAdjuster, F::AbstractArray{Complex,3}, y::Int, x::Int, z::Int) :: SubArray

Compute a 3x3x3 view (SubArray) of the real-DFT `F` considering periodic
(wrap-around) boundary conditions (ie, `F = rfft(s)` for some `s`).
"""
@inline function realDFT_3x3_view(
  S :: FrequencyAdjuster{T, 3}
, F :: AbstractArray{Complex{T},3}
, y :: Int
, x :: Int
, z :: Int
) :: SubArray{Complex{T},3} where T <: AbstractFloat
    # Compute the (y, x, z) index of the real-DFT F considering periodic
    # (wrap-around) boundary conditions.
    @inline function compute_index(y::Int,x::Int,z::Int) :: Tuple{Int,Int,Int}
        x = wraparound(x,S.L[2])
        z = wraparound(z,S.L[3])
        
        if y <= 0
            y = 2-y
            x = x>1 ? S.L[2] - x + 2 : x
            z = z>1 ? S.L[3] - z + 2 : z
        elseif y > S.rfftM
            y = S.rfftM
            x = x>1 ? S.L[2] - x + 2 : x
            z = z>1 ? S.L[3] - z + 2 : z
        end
            
        return (y, x, z) :: Tuple{Int,Int,Int}
    end
    
    # Create LinearIndices object of the window
    LI = LinearIndices(size(F))
    # Store into inds the linear indices of the 3x3x3 region around (y, x, z) after
    # considering periodic boundary conditions.
    inds = [ LI[compute_index(y,x,z)...] for y in y-1:y+1, x in x-1:x+1, z in z-1:z+1 ] :: AbstractArray{Int,3}
    
    # Return a View of F[inds]
    @inbounds return (@view F[inds]) :: SubArray{Complex{T},3}
end

"""
    realDFT_3x3_view(S::FrequencyAdjuster, F::AbstractArray{Complex,N}, p::CartesianIndex{N}) :: SubArray

Compute a 3x3...x3 view (SubArray) of the real-DFT `F` considering periodic
(wrap-around) boundary conditions (ie, `F = rfft(s)` for some `s`).
"""
@inline function realDFT_3x3_view(
  S :: FrequencyAdjuster{T, N}
, F :: AbstractArray{Complex{T},N}
, p :: NTuple{N, Int}
) :: SubArray{Complex{T},N} where T <: AbstractFloat where N
    # Store into buffer the memory to be used
    buffer = Array{Int,1}(undef, 2N+3^N)
    # Then create View of
    # the N-dimensional index v
    v = @view buffer[1:N]
    # the cumulative sizes of the previous dimensions of F
    cumsize = @view buffer[N+1:2N]
    # Store into the first index 1
    cumsize[1] = 1
    for n=2:N
        # Then store into the following cumulative size the value of
        # the previous cumulative size multiplied the previous dimension
        cumsize[n] = cumsize[n-1] * size(F, n-1)
    end
    # And create a view for the indices of the 3x3...x3 region in F around p
    inds = reshape(view(buffer, 2N+1:length(buffer)), ntuple(n -> 3, Val(N)))
    
    # Iterate linearly over the 3x3...x3 region
    @inbounds for i in 1:((3^N))
        # Initialize current index with 1
        ind = 1
        # Compute i-th 0-indexed coordinate of 3x3...x3 region around p
        # and store into v
        v[1] = ((i-1)%3)-1 + p[1]
        for n in 2:N
            # Then iterate over all dimensions of coordinate v
            v[n] = (((i-1)÷3^(n-1))%3)-1 + p[n]
            # and wrap-around the indices
            v[n] = FrequencyAdjustment.wraparound(v[n],S.L)
        end
        
        if v[1] <= 0
            v[1] = 2-v[1]
            for n = 2:N
                v[n] = v[n]>1 ? S.L - v[n] + 2 : v[n]
            end
        elseif v[1] > S.rfftM
            v[1] = S.rfftM
            for n = 2:N
                v[n] = v[n]>1 ? S.L - v[n] + 2 : v[n]
            end
        end
        
        # Compute linear index of coordinate v and store into ind
        for n in 1:N
            ind += (v[n] - 1) * cumsize[n]
        end
        # Store index of i-th linear index into inds
        inds[i] = ind
    end
    
    # Return a View of F[inds]
    @inbounds return (@view F[inds]) :: SubArray{Complex{T},N}
end

"""
    i,j = findpeak(S::FrequencyAdjuster, F::Matrix{Complex})

Find and return the index of the largest value in `abs(F)`, but considering only
indices whose corresponding frequencies are outside the spectral circle ℭ.
"""
function findpeak(
  S::FrequencyAdjuster{T, N}
, F::AbstractArray{Complex{T}, N}
) where T <: AbstractFloat where N
    # maxval stores the maximum value of abs.(F)
    maxval :: T = -one(T)
    # And maxind stores the linear index of maxval
    maxind :: Int = 0
    # Iterate over the indices outside the spectral circle ℭ
    for I in S.indices_outside_spectralcircle
        # Store into absval the squared absolute value of F[I]
        @inbounds absval::T = abs2( F[I] )
        # If absval is greater than current maxval
        if absval > maxval
            # Store absval into maxval
            maxval = absval
            # And store the current index I into maxind
            maxind = I::Int
        end
    end
    # Transform linear index into a tuple with cartesian index and return it
    return ind2sub( size(F), maxind ) :: NTuple{N, Int}
end

""" Perform circular frequency wraparound to the range [-0.5,0.5) """
@inline function freq_wraparound(ξ::T) where T mod(ξ + T(0.5), T(1)) - T(0.5) end

""" Convert a frequency in [-0.5,0.5) to [0,1) by wrapping """
@inline function to_positive_freq_fft(ξ::T) where T ξ < T(0) ? T(1)+ξ-eps(T) : ξ end

""" Convert a frequency in [-0.5,0.5) to [0,0.5] by mirroring """
@inline function to_positive_freq_rfft(ξ) abs(ξ) end

""" Convert frequency in [0,1) to the corresponding (fractional!) DFT index """
@inline function f2idx_fft(S,ξ)  to_positive_freq_fft(ξ) *S.L + 1 end

""" Convert frequency in [0,1) to the corresponding (fractional!) real-DFT index """
@inline function f2idx_rfft(S,ξ) to_positive_freq_rfft(ξ)*S.L + 1 end

""" Prealloc memory for wave detection computation """
function alloc_buffers(S::FrequencyAdjuster{T, N}) where T <: AbstractFloat where N
    # Preallocate memory for wave detection
    buffers = Vector{Array}(undef, 9)
    # First buffer stores s_windowed the samples of the windowed region where waves will be detected
    buffers[1] = Array{T, N}(undef, size(S.windowNd))
    # Second buffer stores S_mn the DFT of the windowed region where the waves will be detected
    # minus the residual Residual of the current iteration
    buffers[2] = Array{Complex{T}, N}(undef, S.rfftM, size(S.windowNd)[2:end]...)
    # Third buffer stores W_mn the residual content in fourier (ie frequency) space
    buffers[3] = similar(buffers[2])
    # Fourth buffer stores the i-th wave values in the windowed region
    buffers[4] = similar(buffers[1], Complex{T})
    # Fifth buffer stores res_windowed the residual content in color space
    buffers[5] = similar(buffers[1])
    # Sixth buffer stores S_orig the initial DFT of the windowed region where the waves will be detected
    buffers[6] = similar(buffers[2])
    # Seventh buffer stores window_shift shifts from the center of the window
    buffers[7] = collect(CartesianIndices(centered(S.windowNd)))
    # Eighth buffer stores helper a buffer for small operations during detection
    buffers[8] = similar(buffers[1])
    return buffers :: Vector{Array}
end

"""
    innerproduct(S::FrequencyAdjuster, A::Matrix{Complex}, B::Matrix{Complex}) :: Complex

Compute the ``l^2`` inner product ⟨a|b⟩ * N, considering that `A = fft(a)` and `B =
rfft(b)`, for some finite sequences `a` and `b`, where `N = length(A)`.
"""
function innerproduct(
  S :: FrequencyAdjuster{T, 2}
, A :: Matrix{Complex{T}}
, B :: Matrix{Complex{T}}
) where T <: AbstractFloat
    # Initialize the inner product AB=⟨a|b⟩ with 0
    AB = zero(Complex{T})

    # Sum the product of A and the complex conjugate of B for half of the window
    for j in 1:S.L, i in 1:S.rfftM
        @inbounds AB += A[i,j] * conj(B[i,j])
    end
    # Sum the product of A and B for the remaining column indices of the last column
    for i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,1] * B[end-i+1,1]
    end
    # Sum the product of A and B with the other half of the window
    for j in 2:S.L, i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,j] * B[end-i+1,end-j+2]
    end

    return AB :: Complex{T}
end

"""
    innerproduct(S::FrequencyAdjuster, A::Array{Complex,3}, B::Array{Complex,3}) :: Complex

Compute the ``l^2`` inner product ⟨a|b⟩ * N, considering that `A = fft(a)` and `B =
rfft(b)`, for some finite sequences `a` and `b`, where `N = length(A)`.
"""
function innerproduct(
  S :: FrequencyAdjuster{T, 3}
, A :: Array{Complex{T},3}
, B :: Array{Complex{T},3}
) where T <: AbstractFloat
    # Initialize the inner product AB=⟨a|b⟩ with 0
    AB = zero(Complex{T})
    
    # Sum the product of A and the complex conjugate of B for half of the window
    for k in 1:S.L, j in 1:S.L, i in 1:S.rfftM
        @inbounds AB += A[i,j,k] * conj(B[i,j,k])
    end
    
    # Sum the product of A and B for the remaining column indices of the last column
    for i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,1,1] * B[end-i+1,1,1]
    end
    # Sum the product of A and B for the remaining column and row indices of the last xy slice
    for j in 2:S.L,i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,j,1] * B[end-i+1,end-j+2,1]
    end
    # Sum the product of A and B for the remaining column and third dimension indices of the last yz slice
    for k in 2:S.L,i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,1,k] * B[end-i+1,1,end-k+2]
    end
    
    # Sum the product of A and B with the other half of the window
    for k in 2:S.L, j in 2:S.L, i in 1:S.rfftM-1
        @inbounds AB += A[S.rfftM+i,j,k] * B[end-i+1,end-j+2,end-k+2]
    end

    return AB :: Complex{T}
end

"""
    innerproduct(S::FrequencyAdjuster, A::Array{Complex,3}, B::Array{Complex,3}) :: Complex

Compute the ``l^2`` inner product ⟨a|b⟩ * N, considering that `A = fft(a)` and `B =
rfft(b)`, for some finite sequences `a` and `b`, where `N = length(A)`.
"""
function innerproduct(
  S :: FrequencyAdjuster{T, N}
, A :: Array{Complex{T},N}
, B :: Array{Complex{T},N}
) where T <: AbstractFloat where N
    # Initialize the inner product AB=⟨a|b⟩ with 0
    AB = zero(Complex{T})
    
    # Sum the product of A and the complex conjugate of B for half (plus last slice half) of the window
    for j in 0:S.L^(N-1)-1, i in 1:S.rfftM
        ind_A = j * S.L + i
        ind_B = j * S.rfftM + i
        @inbounds AB += A[ind_A] * conj(B[ind_B])
    end
    # Sum the product of A and B with the other half (minus first slice of second half) of the window
    for j in 0:S.L^(N-1)-1, i in 2:S.rfftM
        ind = j * S.rfftM + i
        @inbounds AB += A[ S.realDFT_to_DFT_indices[ind] ] * B[ind]
    end
    
    return AB
end

"""
    cis_DFT_to_cos_realDFT!(S::SpectralRemapper, F::Matrix{Complex}, phase)

Convert the DFT `F` of a complex wave to the real-DFT of a real wave with a
given phase shift. The matrix `F` is overwritten with the result. That is:

    f = 0.5 * cis( 2π * (a*x + b*y) )
    g =       cos( 2π * (a*x + b*y + phase) )
    F =  fft(f)
    G = rfft(g)
    cis_DFT_to_cos_realDFT!(S, F, phase)
    @assert G == F[1:rfftM,:]
"""
function cis_DFT_to_cos_realDFT!(
  S :: FrequencyAdjuster{T, 2}
, F :: Matrix{Complex{T}} # This array is overwritten!
, phase :: T
) where T <: AbstractFloat
    fourier_phase = cis(T(2π) * phase) :: Complex{T}
    F .*= fourier_phase
    for j in 1:S.L, i in 1:S.rfftM
        @inbounds F[i,j] += conj(F[ S.realDFT_to_DFT_indices[i,j] ])
    end
    for j in 2:S.rfftM
        @inbounds F[ S.realDFT_to_DFT_indices[1,j] ] = conj(F[1,j])
    end

    nothing # Result is in F[1:S.rfftM,:]
end

"""
    cis_DFT_to_cos_realDFT!(S::FrequencyAdjuster, F::Matrix{Complex}, phase)

Convert the DFT `F` of a complex wave to the real-DFT of a real wave with a
given phase shift. The matrix `F` is overwritten with the result. That is:

    f = 0.5 * cis( 2π * (freqs'X) )
    g =       cos( 2π * (freqs'X + phase) )
    F =  fft(f)
    G = rfft(g)
    cis_DFT_to_cos_realDFT!(S, F, phase)
    @assert all(selectdim(G, 1, i) == selectdim(F, 1, i) for i = 1:rfftM)
"""
function cis_DFT_to_cos_realDFT!(
  S :: FrequencyAdjuster{T, N}
, F :: Array{Complex{T}, N} # This array is overwritten!
, phase :: T
) where T <: AbstractFloat where N
    fourier_phase = cis(T(2π) * phase) :: Complex{T}
    F .*= fourier_phase
    C = CartesianIndices(size(F))
    for j in 0:S.L^(N-1)-1, i in 1:S.rfftM
        ind = j * S.L + i
        @inbounds F[ind] += conj(F[ S.realDFT_to_DFT_indices[C[ind]] ])
    end
    C = CartesianIndices(ntuple(n -> S.rfftM, N))
    for j in 1:S.rfftM^(N-1)-1
        ind = j * S.rfftM + 1
        @inbounds F[ S.realDFT_to_DFT_indices[C[ind]] ] = conj(F[C[ind]])
    end

    nothing # Result is in F[1:S.rfftM,:]
end

"""
    lerp_abs(s::AbstractMatrix{Complex}, αᵥ, αₕ)

Perform a bilinear interpolation inside the 2x2 matrix `m = abs(s)`:

"""
@inline function lerp_abs(s::AbstractVector{Complex{T}}, α::T)::T where T
    @inbounds begin
        # Get corners
        abs_f1::T = abs(s[1])
        abs_f2::T = abs(s[2])
        # Interpolate the points and return the result
        return ( abs_f1 + α*(abs_f2 - abs_f1) ) :: T
    end
end

"""
    lerp_abs(s::AbstractMatrix{Complex}, αᵥ, αₕ)

Perform a bilinear interpolation inside the 2x2 matrix `m = abs(s)`:

    m[1,1] -------- m[1,2]
       |         .     |
       |         .     |
       |         .     |
       |......(αᵥ,αₕ)  |
       |               |
       |               |
    m[2,1] -------- m[2,2]

"""
@inline function lerp_abs(s::AbstractMatrix{Complex{T}}, αᵥ::T, αₕ::T)::T where T
    @inbounds begin
        # Get top and bottom left corners
        abs_f11::T = abs(s[1,1])
        abs_f21::T = abs(s[2,1])
        # Interpolate horizontally the top and bottom left corners with the respective top and bottom right corners
        tophorz::T = abs_f11 + αₕ*(abs(s[1,2]) - abs_f11)
        bothorz::T = abs_f21 + αₕ*(abs(s[2,2]) - abs_f21)
        # Interpolate vertically the interpolated points and return the result
        return ( tophorz + αᵥ*(bothorz - tophorz) ) :: T
    end
end

"""
    lerp_abs(s::AbstractMatrix{Complex}, αᵥ, αₕ, αₚ)

Perform a bilinear interpolation inside the 2x2x2 matrix `m = abs(s)`:

"""
@inline function lerp_abs(s::AbstractArray{Complex{T},3}, αᵥ::T, αₕ::T, αₚ::T) :: T where T
    @inbounds begin
        # For the frontmost slice
        # Get top and bottom left corners
        abs_f11_p1::T = abs(s[1,1,1])
        abs_f21_p1::T = abs(s[2,1,1])
        # Interpolate horizontally the top and bottom left corners with the respective top and bottom right corners
        tophorz_p1::T = abs_f11_p1 + αₕ*(abs(s[1,2,1]) - abs_f11_p1)
        bothorz_p1::T = abs_f21_p1 + αₕ*(abs(s[2,2,1]) - abs_f21_p1)
        # Interpolate vertically the interpolated points
        p1::T = tophorz_p1 + αᵥ*(bothorz_p1 - tophorz_p1)

        # For the backmost slice
        # Get top and bottom left corners
        abs_f11_p2::T = abs(s[1,1,2])
        abs_f21_p2::T = abs(s[2,1,2])
        # Interpolate horizontally the top and bottom left corners with the respective top and bottom right corners
        tophorz_p2::T = abs_f11_p2 + αₕ*(abs(s[1,2,2]) - abs_f11_p2)
        bothorz_p2::T = abs_f21_p2 + αₕ*(abs(s[2,2,2]) - abs_f21_p2)
        # Interpolate vertically the interpolated points
        p2::T = tophorz_p2 + αᵥ*(bothorz_p2 - tophorz_p2)

        # Interpolate the two slices and return the result
        return p1 + αₚ*(p2 - p1) :: T
    end
end

"""
    lerp_abs(s::AbstractMatrix{Complex}, αᵥ, αₕ, αₚ)

Perform a bilinear interpolation inside the 2x2x2 matrix `m = abs(s)`:

"""
@inline function lerp_abs!(s::AbstractVector{T}, α...) :: T where T where N
    @inline function lerp(a, b, α) return a + α * (b - a) end
    current_size = length(s)
    n = 1
    while current_size > 1
        for i in 1:current_size ÷ 2
            s[i] = lerp(s[2i-1], s[2i], α[n])
        end
        current_size = current_size ÷ 2
        n+=1
    end
    return s[1]
end

"""
    lerp_abs(s::AbstractMatrix{Complex}, αᵥ, αₕ, αₚ)

Perform a bilinear interpolation inside the 2x2x2 matrix `m = abs(s)`:

"""
@inline function lerp_abs(s::AbstractArray{Complex{T},N}, αᵥ::T, αₕ::T, αₚ::T, α...) :: T where T where N
    @inline function lerp(a, b, α) return a + α * (b - a) end
    lerp_array = Vector{T}(undef, length(s)÷2)
    for i in 1:length(lerp_array)
        lerp_array[i] = lerp(abs(s[2i-1]), abs(s[2i]), αᵥ)
    end
    return lerp_abs!(lerp_array, αₕ, αₚ, α...)
end

"""
    realDFT_2x2_view(S::FrequencyAdjuster, F::AbstractMatrix{Complex}, fii...) :: SubArray

Compute a 2x2...x2 view (SubArray) of the real-DFT `F` considering periodic
(wrap-around) boundary conditions (ie, `F = rfft(s)` for some `s`).
"""
@inline function realDFT_2x2_view(
  S::FrequencyAdjuster{T, N}
, F::AbstractArray{Complex{T}, N}
, fii...
) :: SubArray{Complex{T},N} where T <: AbstractFloat where N
    # Convert type to Int
    fii = Int.(fii)
    # Map each index to itself and the next index
    # if the index is the last then map to the first index
    I = map((i, L) -> i == L ? [i,1] : [i,i+1], fii, size(F))

    # Return view of indices of the 2x2...x2 region
    @inbounds return (@view F[I...]) :: SubArray{Complex{T},N}
end

"""
    gaussian_window(σ)

Compute an l^2-normalized Gaussian window with an odd number of samples and
clipped at approximately ±4σ.
"""
function gaussian_window(σ; stddevs=4)
    # The length L of the window is given by the standard deviation σ and
    # the number of standard deviations stddevs we want to sample from the
    # center of the window
    L = round(Int, 2*stddevs*σ)
    # If the length is not odd round to the nearest odd number greater than
    # 2*stddevs*σ, ie the length without rounding
    L = isodd(L) ? L : (L > 2*stddevs*σ ? L-1 : L+1)
    # Half the length of the window that divides the window in two equal
    # parts from the center of the window
    P = L ÷ 2
    # Generate the Gaussian window with length L
    h = exp.( -0.5 * (-P:P).^2 / σ^2 )
    # Normalize the Gaussian window
    h = h / norm(h)
end

"""
    compute_dual_window(window::Vector, τ::Int)

Compute a numerical dual window for an overcomplete frame given by copies of
`window` shifted by `τ` pixels.  See Mallat 2009 (2ed), Theorem 5.8.
"""
function compute_dual_window(window::Vector{T}, τ::Int) where T <: AbstractFloat
    # The signal will be analysed with 'window' using shifts of 'τ' samples.
    # One may compute a window which is self-dual by normalizing all the
    # sub-windows, defined by shifts of 'τ', inside 'window'.
    selfdual = copy(window)
    for i = 1:τ
        selfdual[i:τ:end] /= norm(selfdual[i:τ:end])
    end

    # The dual of 'window' is then the multiplicative factor which transforms
    # the synthesis and analysis with 'window' and 'dual', respectively, onto
    # the synthesis and analysis with the 'selfdual':
    #
    #   window ⋅ dual = selfdual ⋅ selfdual,
    #
    # and thus:
    dual = selfdual.^2 ./ window

    return dual :: Vector{T}
end

"""
    isnegativedefinite(Q::Symmetric)

Test wheter the symmetric matrix Q is negative definite.
"""
@inline function isnegativedefinite(Q::Symmetric)
    isposdef(-Q)
end

"""
    find_freqs_and_amplitude(S, S_mn)

Given a signal S and its real DFT S_mn, compute the instantaneous frequencies
and amplitude of the wave with highest amplitude.
"""
function compute_freqs_and_amplitude(S::FrequencyAdjuster{T, N}, S_mn::Array{Complex{T}, N}) where T <: AbstractFloat where N
    # Find 2D index of the maximum
    peak = findpeak(S, S_mn)

    # Fit a quadratic surface to the neighborhood of the maximum. We fit
    # the quadratic to the log of the magnitude of the spectrum since
    # the log of the Gaussian is exactly a quadratic function.
    S_view::SubArray{Complex{T},N} = realDFT_3x3_view(S, S_mn, peak)
    Q, L, C = quadratic_lsqfit_log_abs(S_view)

    # Check negative definiteness of the quadratic
    if !isnegativedefinite(Q)
        # Quadratic is not negative definite and has no maximum. Return a 0 amplitude wave
        return 0, 0
    end

    # Find the exact location of the quadratic's maximum.
    max_coord, max_val::T = quadratic_critical_point(Q, L, C)

    if any(x -> !(0 <= x <= 4), max_coord)
        # The maximum falls outside the 3x3 neighborhood of the peak. Return a 0 amplitude wave
        return 0, 0
    end

    # Map indices of the 3x3 view to indices of the full spectrum S_mn.
    max_coord = map((x, y) -> x + y - 2, peak, max_coord)

    # Compute horizontal and vertical frequencies in [-0.5,0.5)
    freqs = map((x) -> T(x-1)/S.L |> freq_wraparound, max_coord)

    # Compute the amplitude, we must multiply by 2 as a correction due to
    # complex-conjugate pairs (the signal is real).
    amplitude::T = exp(max_val) # We fitted the quadratic to the log of the spectrum.
    amplitude = 2amplitude / sum(S.windowNd)

    # Compute Eq.(17) in the SR paper. The magnitude of the DFT ĝ of the
    # Gaussian window g is also a gaussian with std.dev. Σ = 1/(2π σ).
    total = one(T)
    freqs_vec = [freqs...] # conversion from Tuple to Array
    dist = zeros(N)
    for i in 0:((3^N)-1)
        map!((freq, n) -> exp(-0.5*(((-2 * freq) + ((i÷3^(n-1))%3)+1)^2) / S.Σ^2), dist, freqs_vec, 1:N)
        total += prod(dist)
    end
    amplitude *= inv(total)
    
    return freqs, amplitude
end

"""
    detect_local_waves(S::FrequencyAdjuster, s::AbstractMatrix, I::CartesianIndex)

Decompose the real-DFT spectrum of `s` as a sum of non-harmonic waves that lie
outside the spectral circle ℭ, where `s` is a local portion of a larger image,
centered at `I`. Return `\\mathcal W_{M}^\\varphi`, for `M` = `I`.
"""
function detect_local_waves(
  S :: FrequencyAdjuster{T, N}
, s :: AbstractArray{T, N}
, I :: NTuple{N, Integer}
; RFFT  :: AbstractFFTs.Plan{T} = plan_rfft(S.windowNd)
, IRFFT :: AbstractFFTs.Plan{Complex{T}} = inv(RFFT)
, FFT   :: AbstractFFTs.Plan{Complex{T}} = plan_fft!( Matrix{Complex{T, N}}(undef,size(S.windowNd)) )
, buffers :: Vector{Array} = alloc_buffers(S)
, max_waves :: Integer = 10
) where T <: AbstractFloat where N
    # Pre-allocated memory
    s_windowed   = buffers[1] :: Array{T, N}
    S_mn         = buffers[2] :: Array{Complex{T}, N}
    Residual     = buffers[3] :: Array{Complex{T}, N}
    W_mn         = buffers[4] :: Array{Complex{T}, N}
    res_windowed = buffers[5] :: Array{T, N}
    S_orig       = buffers[6] :: Array{Complex{T}, N}
    window_shift = buffers[7]
    helper       = buffers[8] :: Array{T, N}

    # Perform windowing of the signal with the Gaussian.
    s_windowed .= s .* S.windowNd

    # Compute the mean-centered signal s̄.
    meanval = sum(s_windowed) / sum(S.windowNd)
    s_centered_windowed = s_windowed
    s_centered_windowed .-= meanval .* S.windowNd

    # Compute S_mn, the DFT of s
    mul!(S_mn, RFFT, s_centered_windowed)

    # Copy S_mn
    Residual .= S_mn
    S_orig .= S_mn

    # Loop until the amplitude of detected waves falls below a threshold or for
    # a maximum of max_waves detected waves
    waves = Vector{Wave{T, N}}()
    largest_wave_amplitude :: T = zero(T)
    for iterations in 1:max_waves

        freqs, amplitude = compute_freqs_and_amplitude(S, S_mn)

        # Check if the amplitude falls below the threshold
        if amplitude < (1/4) * largest_wave_amplitude
            # Amplitude is too small. Continue.
            break
        end

        # Update the maximum amplitude so far
        largest_wave_amplitude = max(amplitude, largest_wave_amplitude)

        # Build zero-phase wave W_mn for computing Eq.(15) of the SR paper
        l = S.L ÷ 2
        broadcast!((δ) -> T(2π) * sum(freqs.*(I.+δ.I)), helper, window_shift)
        W_mn .= @. amplitude * cis(helper) * T(0.5) * S.windowNd
        FFT*W_mn # in-place FFT

        # Solve for the phase shift of the wave using Eq.(15) of the SR paper
        WS::Complex{T} = innerproduct(S, W_mn, S_mn)
        F = c′ -> (cos(2T(π)*c′)*real(WS) - sin(2T(π)*c′)*imag(WS))::T
        c1::T,c2::T = [ atan( -imag(WS), real(WS) ), atan( imag(WS), -real(WS) ) ] / 2π
        phase::T = F(c1) > F(c2) ? c1 : c2
        phase = mod(phase, 1)

        # Convert DFT (from fft) to a real-DFT (from rfft)
        cis_DFT_to_cos_realDFT!(S, W_mn, phase)
        W_view = @view W_mn[1:S.rfftM, ntuple(n -> :, Val(N-1))...]

        # Compute frequency-remapping location using φ
        ρ = norm(freqs)
        ρ_remapped = T(0.4) / S.R
        freqs_remapped = normalize([freqs...]) * ρ_remapped

        # Remove the wave from the spectrum used for peak detection.
        energybefore = sum(abs2, S_mn)
        S_mn .-= W_view
        energyafter = sum(abs2, S_mn)

        if energyafter >= energybefore
            # Highly-nonorthogonal decomposition (the energy did not decrease). Continue.
            break
        end

        # Compute inner products for the ensemble inequality (8) of the SR paper.
        # To compute the magnitude of the non-harmonic Fourier coefficient
        # C = ⟨s̄ₘₙ⋅gₘₙ|fᵩ⟩, we perform a bilinear interpolation in the
        # magnitude of the DFT of s̄ₘₙ⋅gₘₙ.

        # Indices for the bilinear interpolation.
        firf = @view helper[1:N]
        firi = @view helper[N+1:2N]
        firf[1],firi[1] = modf( f2idx_rfft(S,freqs_remapped[1]) )
        for n in 2:N
            firf[n], firi[n] = modf(f2idx_fft( S, freqs_remapped[n]))
        end
        S_view = realDFT_2x2_view(S, S_orig, firi...) # Select 2x2 view for interpolation
        C = lerp_abs( S_view, firf...)

        window_energy::T = sum(S.windowNd)

        # Compute the amplitude for inequality (8). We must divide by 2 as a
        # correction due to complex-conjugate pairs (the signal is real).
        α = amplitude * T(0.5)

        # Perform selective frequency remapping
        if α <= T(10^0.06) / window_energy * C
            # We do not adjust this wave since it belongs to an harmonic ensemble.
        elseif ρ <= S.spectralradius
            # We do not adjust this wave since it is inside the spectral circle ℭ.
        else
            # This wave will be adjusted. Add it to \mathcal W^\varphi.
            push!( waves, Wave(freqs,phase,amplitude) )

            # Remove the wave from the residual spectrum (ie, add it to \mathcal W^{!\varphi})
            Residual .-= W_view
        end
    end

    # Inverse real-DFT of the residual
    mul!(res_windowed, IRFFT, Residual)

    # Add back the "0-th harmonic" (DC component)
    res_windowed .+= meanval .* S.windowNd

    return (res_windowed, waves) :: Tuple{ Array{T,N}, Vector{Wave{T, N}} }
end

"""
    padsize(siz::NTuple, P::Int) = map(λ->λ+4P, siz)

Compute the necessary size for storing a `4P`-padded version of an array of original size `siz`.
"""
padsize(siz::NTuple{N,Int}, P::Int) where N = map(λ->λ+4P, siz)

"""
    padsize(siz::NTuple, P::Vector{Int}) = siz.+4P

Compute the necessary size for storing a `4P`-padded version of an array of original size `siz`.
"""
padsize(siz::NTuple{N,Int}, P::Vector{Int}) where N = Tuple(siz.+4P)

"""
    padsize(src::AbstractArray, P::Int)

Compute the necessary size for storing a `4P`-padded version of array `src`.
"""
padsize(src::AbstractArray, P::Int) = padsize(size(src), P)

"""
    padsize(src::AbstractArray, P::Vector{Int})

Compute the necessary size for storing a `4P`-padded version of array `src`.
"""
padsize(src::AbstractArray, P::Vector{Int}) = Tuple(padsize(size(src), P))

"""
    mirrorpad!(dst::AbstractArray{T}, src::AbstractArray{T}, P::Int; fillvalue = zero(T))

Create a mirror-padded copy of `src` in `dst`.
"""
function mirrorpad!(dst::AbstractArray{T}, src::AbstractArray{T}, P::Int) where T

    @assert size(dst) == padsize(src, P) "Size of dst does not match size of src and pad length P."

    I = ntuple(x->(1+2P):(2P+size(src,x)), ndims(src))
    dst[ I... ] = src

    for d in 1:ndims(src)
        # Prologue mirror
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (1):(2P)
                            elseif x > d
                                return (1 + 2P):(1 + 2P + size(src,x) - 1)
                            end
                        end, ndims(src))
        J = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (2P + 2P):-1:(2P + 1)
                            elseif x > d
                                return (1 + 2P):(1 + 2P + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] = dst[ J... ]

        # Epilogue mirror
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (1 + 2P + size(src,x)):(1 + 2P + size(src,d) + 2P - 1)
                            elseif x > d
                                return (1 + 2P):(1 + 2P + size(src,x) - 1)
                            end
                        end, ndims(src))
        J = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (2P + size(src,x)):-1:(2P + size(src,x) - 2P + 1)
                            elseif x > d
                                return (1 + 2P):(1 + 2P + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] = dst[ J... ]
    end

    return dst
end

"""
    mirrorpad!(dst::AbstractArray{T}, src::AbstractArray{T}, P::Int; fillvalue = zero(T)) where T

Create a mirror-padded copy of `src` in `dst`, with an additional frame filled with `fillvalue` values.
"""
function mirrorpad!(dst::AbstractArray{T}, src::AbstractArray{T}, P::Vector{Int}; fillvalue = zero(T)) where T

    @assert size(dst) == padsize(src, P) "Size of dst does not match size of src and pad length P."

    I = ntuple(x->(1+2P[x]):(2P[x]+size(src,x)), ndims(src))
    dst[ I... ] .= src

    for d in 1:ndims(src)
        # Prologue zero
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return 1:P[x]
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] .= fillvalue
        
        # Epilogue zero
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (1 + 2P[x] + size(src,x) + P[x]):size(dst,x)
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] .= fillvalue

        # Prologue mirror
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (1 + P[x]):(2P[x])
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        J = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (2P[x] + P[x]):-1:(2P[x] + 1)
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] = dst[ J... ]

        # Epilogue mirror
        I = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (1 + 2P[x] + size(src,x)):(1 + 2P[x] + size(src,d) + P[x] - 1)
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        J = ntuple(  x->begin
                            if x < d
                                return Colon()
                            elseif x == d
                                return (2P[x] + size(src,x)):-1:(2P[x] + size(src,x) - P[x] + 1)
                            elseif x > d
                                return (1 + 2P[x]):(1 + 2P[x] + size(src,x) - 1)
                            end
                        end, ndims(src))
        dst[ I... ] = dst[ J... ]
    end

    return dst
end


"""
    detect_waves!(S::FrequencyAdjuster, s::AbstractMatrix)

Decompose the signal `s` using Gabor analysis. For a series of overlapping
Gaussian windows centered at pixels `(m,n)`, decompose the spectrum of the
windowed portion of `s` as a sum of nonharmonic waves, ``\\mathcal W_{mn}``.

Returns `-1` if `s` has no elements
"""
function detect_waves!(
  S :: FrequencyAdjuster{T, N}
, s :: AbstractArray{T, N}
) where T <: AbstractFloat where N
    @assert isodd(S.L) "This code has been written for odd-sized windows"
    # If the input is empty
    if isempty(s)
        # Set detection data to empty arrays too
        S.data[:q] = T[]
        S.data[:waves] = Vector{Wave{T, N}}[]
        # And return -1 to indicate that there is no wave to detect
        return -1
    end

    P = S.L÷2 # Signal padding

    # Pad the signal with mirrored boundary elements
    s_with_pad = Array{T}(undef, padsize(s,P))
    mirrorpad!(s_with_pad, s, P)

    # Pre-plan fast FFT
    FFTW.set_num_threads(1)

    FFT = plan_fft!(Array{Complex{T}, N}(undef,size(S.windowNd)))
    RFFT = plan_rfft(S.windowNd)
    IRFFT = inv(RFFT)

    # Indices in each dimension of the τ sparsed samples in the padded image
    indices = [1:S.τ:2P + size(s,i) for i=1:N]
    indices_length = length.(indices)

    # Alloc vector of waves, \mathcal W_{m} (for all m)
    waves = Array{ Vector{Wave{T, N}}, N }(undef, indices_length...)

    # Alloc output array for the residual image q
    q = zeros(T, size(s_with_pad) )

    # Alloc intermediary computation buffers
    buffers = Dict(i => alloc_buffers(S) for i in 1:Threads.nthreads())

    mutex = Threads.SpinLock()
    count = Threads.Atomic{Int}(0)
    dims_tuple = ntuple(n -> n, Val(N))
    @withprogress name="Detecting local waves..........." begin
        total = indices_length[end]
        offsets = cumprod([1, indices_length[1:end-2]...])
        Threads.@threads for (i_n) in collect(1:indices_length[end])
            params = Dict(
                :FFT     => FFT,
                :RFFT    => RFFT,
                :IRFFT   => IRFFT,
                :buffers => buffers[Threads.threadid()],
                :max_waves => S.max_waves
            )
            for i = 1:prod(indices_length[1:end-1])
                # Extract LxL...xL window and detect waves. The elementwise product with
                # the Gaussian is performed inside the function detect_local_waves().
                wave_ind = CartesianIndex(map((n) -> ((i-1) ÷ offsets[n] % indices_length[n])+1, 1:length(offsets))..., i_n)
                I = map(i -> indices[i][wave_ind[i]], dims_tuple)

                @inbounds s_local = @view s_with_pad[map((x) -> x:x+S.L-1, I)...]
                res, waves[wave_ind] = detect_local_waves(S, s_local, I.-P; params...)

                # Produce residual image (where detected waves have been removed)
                lock(mutex) do
                    @inbounds q[map((x) -> x:x+S.L-1, I)...] .+= res .* S.dualwindowNd
                end
            end

            count[] += 1 # Atomic counter
            if Threads.threadid() == 1
                @logprogress count[] / total
            end
        end
    end

    # Remove padding from the residual image and waves
    @inbounds S.data[:q] = q[map((x) -> 2P+1:2P+x, size(s))...]
    @inbounds S.data[:waves] = SpacedArray(waves, map((x) -> x .- P, indices)...)
end
