### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 861aac40-552b-11eb-1151-6b74f5ed3de5
# Used Packages Index
begin
	using LinearAlgebra
end ;

# ╔═╡ 2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
md"""
## Notebook report: Geometric multigrid in 2D
"""

# ╔═╡ 9c806a14-551d-11eb-0676-5942da7c95e7
# Constants & Parameters Setting
begin
	n 		= 4
	h 		= 1 / n
	ω 		= 1.5
	σ 		= 0.75
	ugrid 	= zeros(n, n)
	u 		= ugrid'[:]
end ;

# ╔═╡ 5c93118a-5555-11eb-14b1-872b186d3b0b
"""
    boundaries(grid::Array{Float64,2})

A function that inject the boundaries conditions into the grid.
To be used after each computing iteration towards convergence.
"""
function boundaries(grid)
	n 	= size(grid, 1)		# U is always square in our case
	grid[1, 1:end] 		.= 0
	grid[1:end, 1] 		.= 0
	grid[end, 1:end] 	.= 0
	grid[1:end, end] 	.= 0
	return u
end ;

# ╔═╡ c33cb002-553d-11eb-31d3-3176668c06df
md"""
**Constructing NxN Grid Laplacian 2D-operator**
"""

# ╔═╡ c68281e6-5539-11eb-10c0-fd665be52ae0
html"""
<p style="text-align: center;">
$A u = \Delta u$ = $\frac{∂²u}{∂x²}\ e_x$ + $\frac{∂²u}{∂y²}\ e_y$
</p>
<center><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQd4MNLEj5e4xld1062xPxR-xPJ13Q7QXVQSg&usqp=CAU" alt="Operator A for Au = f"></center>
"""

# ╔═╡ 34f711ea-554f-11eb-342f-0358c848c965
md"""
We then need to transform our differential equation into a linear problem in the form of $\textbf{Au=f}$.
"""

# ╔═╡ 207fd796-5550-11eb-2e66-a93e8f66e338
md"""
For that, we only have to use the matrix $σI$ to then factorize the equation by $u(x,y)$
"""

# ╔═╡ ba523a0e-554f-11eb-162f-8b6e6434d21c
html"""
<p style="text-align: center;">
$-Δ u(x,y) + σu(x,y) = f(x,y)$<br/>
$(σI-Δ)u(x,y) = f(x,y)$
</p>
"""

# ╔═╡ be97faee-5532-11eb-1466-15d0f84888cf
# Operators Setting
begin
	∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
	∂x² = (1 / h^2) * kron(∂², I(n))	# kron is the Kronecker product
	∂y² = (1 / h^2) * kron(I(n), ∂²)	# h is the unit of displacement
	Δ 	= ∂x² + ∂y²
	A 	= σ * I(size(Δ, 1)) - Δ
end ;

# ╔═╡ 02c83b80-5546-11eb-2fd1-09fc47ed0d66
"""
    SOR(A, b, ω, u₀, ϵ, maxiter)

Successive Over Relaxation iterative smoother.
"""
function SOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 1000000)
    u = u₀; iter = 1
	M = Diagonal(A) / ω + UnitLowerTriangular(A) - I(size(A, 1))
	while iter <= maxiter
		r = b - A*u
		z = inv(M) * r
		u += z
		(norm(r, 2) > ϵ) || break 	# Convergence check
		iter += 1
	end
	return u
end ;

# ╔═╡ 1d22a786-5555-11eb-2c5f-b913a5153bbc
a = SOR(A, 2.5721413*ones(n*n), ω, 124314*ones(n*n), 1e-7, 2000)

# ╔═╡ b3407b32-5558-11eb-0351-5b250c8d44bc
A * a

# ╔═╡ Cell order:
# ╟─2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
# ╠═861aac40-552b-11eb-1151-6b74f5ed3de5
# ╠═9c806a14-551d-11eb-0676-5942da7c95e7
# ╠═5c93118a-5555-11eb-14b1-872b186d3b0b
# ╟─c33cb002-553d-11eb-31d3-3176668c06df
# ╟─c68281e6-5539-11eb-10c0-fd665be52ae0
# ╟─34f711ea-554f-11eb-342f-0358c848c965
# ╟─207fd796-5550-11eb-2e66-a93e8f66e338
# ╟─ba523a0e-554f-11eb-162f-8b6e6434d21c
# ╠═be97faee-5532-11eb-1466-15d0f84888cf
# ╠═02c83b80-5546-11eb-2fd1-09fc47ed0d66
# ╠═1d22a786-5555-11eb-2c5f-b913a5153bbc
# ╠═b3407b32-5558-11eb-0351-5b250c8d44bc
