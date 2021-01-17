using LinearAlgebra
using SparseArrays
using Plots

function A₁(n::Int, σ::Float64)
    ∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
    ∂x² = (n^2) * kron(sparse(∂²), I(n))
    ∂y² = (n^2) * kron(I(n), sparse(∂²))
    Δ 	= ∂x² + ∂y²
    return σ * I(n^2) - Δ
end

function A₂(n::Int, ϵ::Float64)
    ∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
    ∂x² = (n^2) * kron(sparse(∂²), I(n))
    ∂y² = (n^2) * kron(I(n), sparse(∂²))
    return - ∂x² - ϵ * ∂y²
end

A₁(σ::Float64) = n -> A₁(n, σ)
A₂(ϵ::Float64) = n -> A₂(n, ϵ)

function boundaries(v)
    n = Int(sqrt(size(v,1)))
    gridᵥ = reshape(v, (n, n))'
    gridᵥ[1,:] .= 0
    gridᵥ[end,:] .= 0
    gridᵥ[:,1] .= 0
    gridᵥ[:,end] .= 0
    return gridᵥ |> transpose |> vec
end

function Jacobi(A, b, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10, bounds = false)
    u = u₀; n = Int(sqrt(size(A, 1))); iter = 0
    M = Diagonal(A)
    N = UnitLowerTriangular(A) + UnitUpperTriangular(A) - 2*I(n^2)
    while iter <= maxiter
        iter += 1
        if bounds
                u = boundaries(u)
        end
        u = inv(M) * (N*u + b)
        (norm(b - A*u, 2) > ϵ) || break
    end
    return u
end

function JOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10, bounds = false)
    u = u₀; iter = 0
    M = Diagonal(A) / ω
    while iter <= maxiter
        iter += 1
        if bounds
                u = boundaries(u)
        end
        r = b - A*u
        z = inv(M) * r
        u += z
        (norm(r, 2) > ϵ) || break
    end
    return u
end

function SOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10, bounds = false)
    u = u₀; n = Int(sqrt(size(A, 1))); iter = 0
    D = Diagonal(A)
    L = UnitLowerTriangular(A) - I(n^2)
    U = UnitUpperTriangular(A) - I(n^2)
    while iter <= maxiter
            iter += 1
            if bounds
                    u = boundaries(u)
            end
            u = inv(D + ω * L) * (ω * b - (ω * U + (ω-1) * D) * u)
            (norm(b - A*u, 2) > ϵ) || break
    end
    return u
end

# Restriction
injection(grid) = grid[2:2:end, 2:2:end]

function halfweight(grid)
    g = Float64.(grid)
    for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
        g[i,j] = g[i,j] / 2 - (g[i-1,j] + g[i+1,j] +
                               g[i,j-1] + g[i,j+1]) / 8
    end
    return injection(g)
end

function ϵ_halfweight(ϵ, grid)
    g = Float64.(grid)
    for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
        g[i,j] = ϵ * g[i,j] / 4 - (
                g[i-1,j] + g[i+1,j]
                + (1 - ϵ/2) * g[i,j-1] + (1 - ϵ/2) * g[i,j+1]) / 8
    end
    return injection(g)
end

ϵ_halfweight(ϵ) = grid -> ϵ_halfweight(ϵ, grid)

function fullweight(grid)
    g = Float64.(grid)
    for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
        g[i,j] = g[i,j] / 4 - (g[i-1,j] + g[i+1,j] +
            g[i,j-1] + g[i,j+1]) / 8 - 
            (g[i-1,j-1] + g[i+1,j+1] +
             g[i-1,j+1] + g[i+1,j-1]) / 16
    end
    return injection(g)
end

# Prolongation
function enlarge(grid)
    n = size(grid,1) * 2
    n == 2 && return repeat(grid, n, n)
    g = zeros((n,n))
    for i=2:n-1, j=2:n-1
        g[i, j] = grid[i÷2, j÷2]
    end
    return g
end

function linearize(grid)
    n = size(grid,1) * 2
    n == 2 && return repeat(grid, n, n)
    g = zeros((n,n))
    for i=2:n-1, j=2:n-1
        g[i, j] = (grid[Int(floor((i+1)/2)), Int(floor((j+1)/2))] 
                    + grid[Int(ceil((i+1)/2)), Int(floor((j+1)/2))] 
                    + grid[Int(floor((i+1)/2)), Int(ceil((j+1)/2))] 
                    + grid[Int(ceil((i+1)/2)), Int(ceil((j+1)/2))]) / 4
    end
    return g
end

function multigrid(A, b, u, l, ω, ϵ=1e-7, steps=1,
		restrict=injection, prolong=enlarge, iter=10)
    n  = Int(sqrt(size(b, 1)))
    Aₙ = A(n)
    if l == 0
        # We can also use a direct solver instead
        # u = Array(Aₙ) \ b
        u = JOR(Aₙ, b, ω, u, ϵ, iter)			# Resolution
    else
        u = JOR(Aₙ, b, ω, u, ϵ, iter)			# Pre-smoothing

        # Defect restriction
        r = reshape(b - Aₙ*u, (n, n)) |> transpose |>
                restrict |> transpose |> vec

        # Coarse-level Correction
        δᵣ = zeros(size(r))
        for i=1:steps
            δᵣ = multigrid(A, r, δᵣ, l-1, ω, ϵ, steps,
                           restrict, prolong, iter*3)
        end

        # Defect Prolongation δᵣ → δ
        δ = reshape(δᵣ, (n÷2, n÷2)) |> transpose |>
                prolong |> transpose |> vec
        
        u += δ 						# Correction
        u = JOR(Aₙ, b, ω, u, ϵ, iter)			# Post-smoothing
    end
    return u
end ;

