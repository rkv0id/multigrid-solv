### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 861aac40-552b-11eb-1151-6b74f5ed3de5
# Used Packages Index
begin
	using LinearAlgebra 	# Linear Algebra
	using Plots 			# Plotting & Visualization
end ;

# ╔═╡ 2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
md"""
## Notebook report: Geometric multigrid in 2D
"""

# ╔═╡ 9c806a14-551d-11eb-0676-5942da7c95e7
# Resolution parameters Setting
begin
	n 		= 64
	h 		= 1 / n
	ω 		= 0.5
	σ 		= 0.75
	ϵ 		= 0.4
end ;

# ╔═╡ c33cb002-553d-11eb-31d3-3176668c06df
md"""
**Constructing the Laplacian 2D-operator and the Problems PDEs**
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
$-Δ u(x,y) + σu(x,y) = f(x,y)\ \ \ \ (P_1)$<br/>
$(σI-Δ)u(x,y) = f(x,y)$<br/>
$A_1=σI-Δ$<br/>
</p>
"""

# ╔═╡ 4f335f12-573b-11eb-042c-a97c20dd929d
md"""
As for the Anisotropic problem $P_2$, the construction is similar but only using the partial derivatives
"""

# ╔═╡ 8a710356-573b-11eb-11b0-1b418859bb13
html"""
<p style="text-align: center;">
$-\frac{\partial^2 u(x,y)}{∂x²} - \epsilon \frac{\partial^2 u(x,y)}{∂y²} = f(x,y)\ \ \ \ (P_2)$<br/>
$A_2=(\frac{\partial^2}{∂x²} - \epsilon \frac{\partial^2}{∂y²})I$<br/>
</p>
"""

# ╔═╡ be97faee-5532-11eb-1466-15d0f84888cf
# PDEs Operators Setting
begin
	"""
		A₁(n)

	A function that returns the Poisson probem's linear operator.
	"""
	function A₁(n)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (1 / h^2) * kron(∂², I(n))	# kron is the Kronecker product
		∂y² = (1 / h^2) * kron(I(n), ∂²)	# h is the unit of displacement
		Δ 	= ∂x² + ∂y²
		return σ * I(n^2) - Δ
	end

	"""
		A₂(n)

	A function that returns the Anisotropic probem's linear operator.
	"""
	function A₂(n)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (1 / h^2) * kron(∂², I(n))	# kron is the Kronecker product
		∂y² = (1 / h^2) * kron(I(n), ∂²)	# h is the unit of displacement
		return - ∂x² - ϵ * ∂y²
	end
end ;

# ╔═╡ d3ecaa6c-55ef-11eb-032c-632239a2f100
md"""
Our grid boundaries are nulled by the equation's initial condition as $u(x,y)=0\ on\ \partial \Omega$ which means that we should reinject this boundaries condition after each smoothing operation.
"""

# ╔═╡ 5c93118a-5555-11eb-14b1-872b186d3b0b
"""
    boundaries(grid::Array{Float64,2})

A function that inject the boundaries conditions into the grid.
To be used after each computing iteration towards convergence.
"""
function boundaries(grid)
	g = copy(grid)
	g[1, 1:end] 	.= 0
	g[1:end, 1] 	.= 0
	g[end, 1:end] 	.= 0
	g[1:end, end] 	.= 0
	return grid
end ;

# ╔═╡ bbfb0164-5771-11eb-09c0-8d8bbf8e7434
"""
    Jacobi(A, b, u₀, ϵ, maxiter)

Jacobi iterative smoother.
"""
function Jacobi(A, b, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10)
    u = u₀; iter = 0
	M = Diagonal(A)
	N = UnitLowerTriangular(A) + UnitUpperTriangular(A) - 2*I(size(A, 1))
	while iter <= maxiter
		iter += 1
		u = inv(M) * (N*u + b)
		(norm(b - A*u, 2) > ϵ) || break 	# Convergence check
	end
	return u
end ;

# ╔═╡ aafcba2a-5750-11eb-19e9-8fa5b1952b54
"""
    JOR(A, b, ω, u₀, ϵ, maxiter)

Jacobi Over Relaxation iterative smoother.
"""
function JOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10)
    u = u₀; iter = 0
	M = Diagonal(A) / ω
	while iter <= maxiter
		iter += 1
		r = b - A*u
		z = inv(M) * r
		u += z
		(norm(r, 2) > ϵ) || break 	# Convergence check
	end
	return u
end ;

# ╔═╡ 02c83b80-5546-11eb-2fd1-09fc47ed0d66
"""
    SOR(A, b, ω, u₀, ϵ, maxiter)

Successive Over Relaxation iterative smoother.
"""
function SOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10)
    u = u₀; iter = 0
	M = Diagonal(A) / ω + UnitLowerTriangular(A) - I(size(A, 1))
	while iter <= maxiter
		iter += 1
		r = b - A*u
		z = inv(M) * r
		u += z
		(norm(r, 2) > ϵ) || break 	# Convergence check
	end
	return u
end ;

# ╔═╡ 40bdad12-55fa-11eb-0b00-c57014be1cdf
# Multigrid's Interpolation operators
begin
	# Restriction
	injection(grid) = grid[2:2:end, 2:2:end]
	
	function halfweight(grid)
		g = Float64.(grid)
		for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
			g[i,j] = g[i,j] / 2 - (
				g[i-1,j] + g[i+1,j]
				+ g[i,j-1] + g[i,j+1]) / 8
		end
		return injection(g)
	end
	
	function fullweight(grid)
		g = Float64.(grid)
		for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
			g[i,j] = g[i,j] / 4 - (
				g[i-1,j] + g[i+1,j]
				+ g[i,j-1] + g[i,j+1]) / 8 - (
				g[i-1,j-1] + g[i+1,j+1]
				+ g[i-1,j+1] + g[i+1,j-1]) / 16
		end
		return injection(g)
	end

	# Prolongation
	function enlarge(grid)
		n = size(grid,1) * 2 				# Working with square domains only
		n == 2 && return repeat(grid, n, n)	# size(grid) == (1,1)
		g = zeros((n,n))
		for i=2:n-1, j=2:n-1
			g[i, j] = grid[i÷2, j÷2]
		end
		return g
	end
	
	function linearize(grid)
		n = size(grid,1) * 2 				# Working with square domains only
		n == 2 && return repeat(grid, n, n)	# size(grid) == (1,1)
		g = zeros((n,n))
		for i=2:n-1, j=2:n-1
			g[i, j] = (grid[Int(floor((i+1)/2)), Int(floor((j+1)/2))] 
						+ grid[Int(ceil((i+1)/2)), Int(floor((j+1)/2))] 
						+ grid[Int(floor((i+1)/2)), Int(ceil((j+1)/2))] 
						+ grid[Int(ceil((i+1)/2)), Int(ceil((j+1)/2))]) / 4
		end
		return g
	end
end ;

# ╔═╡ 846b2244-56ac-11eb-3b62-e9b1b45d3582
"""
    multigrid(A, b, u, l, ω, restrict, prolong,
			  iterpre, iterpost, steps)

l-level Multigrid cycle.
	- A: the PDE's linear operator constructor
	- restrict: function for grid restriction
	- prolong: function for grid prolongation
	- iterpre: number of iterations per presmoothing
	- iterpost: number of iterations per postsmoothing
	- steps: number of multigrid callbacks per cycle:
		- 1 (default) is for V-cycle
		- 2 is for normal W-cycle
		- n (> 2) is for W-cycles with n sub-refinements
"""
function multigrid(A, b, u, l, ω, ϵ=1e-7, steps=1,
		restrict=injection, prolong=linearize,
		iterpre=30, iterpost=30)
	n  = Int(sqrt(size(b, 1)))
	Aₙ = A(n)
	u  = JOR(Aₙ, b, ω, u, ϵ, iterpre)			# Pre-smoothing
	d  = b - Aₙ*u 								# Defect computing	

	# Defect restriction
	g  = reshape(d, (n, n))'
	gₛ = restrict(g)
	dₛ = gₛ'[:]

	# Sub-level resolution
	if l == 1
		Aₛ = A(n÷2)
		# vₛ = Aₛ \ dₛ
		vₛ = JOR(Aₛ, dₛ, ω, zeros(size(dₛ)), ϵ, iterpre + iterpost)
	else
		vₛ = multigrid(A, dₛ, zeros(size(dₛ)),
				l-1, ω, ϵ, steps, restrict,
				prolong, iterpre, iterpost)
		for i=2:steps
			vₛ = multigrid(A, dₛ, vₛ,
				l-1, ω, ϵ, steps, restrict,
				prolong, iterpre, iterpost)
		end
	end
	
	# Defect Prolongation
	gₛ = reshape(vₛ, (n÷2, n÷2))'
	g  = prolong(gₛ)
	v  = g'[:]
	u  += v 									# Update approximation
	u  = JOR(Aₙ, b, ω, u, ϵ, iterpost)			# Post-smoothing
	u  = boundaries(reshape(u, (n, n))')'[:]	# Re-injecting boundaries conditions
	return u
end ;

# ╔═╡ c487b42c-55ea-11eb-0af0-2d7b1bcced1f
# Comparing solvers convergence for the 2D-Poisson Problem
begin
	A = A₁
	u = zeros(n^2)
	b = [sin(i*j*π/15) for i=1:n for j=1:n]

	# JOR
	u₁ = JOR(A(n), b, ω, u, 1e-30, 500)
	a₁ = b - A(n)*u₁
	g₁ = reshape(a₁, (n, n))'
	htmp₁ 	= heatmap(1:n, 1:n, g₁, fmt=:png, ratio=1,
						title="SOR Solution Error - Heatmap")
	sfc₁ 	= surface(g₁, fmt=:png, ratio=1,
						legend=false, title="SOR Solution Error - Surface")

	# Multigrid
	u₂ = multigrid(A, b, u, 5, ω, 1e-30, 4)
	a₂ = b - A(n)*u₂
	g₂ = reshape(a₂, (n, n))'
	htmp₂ 	= heatmap(1:n, 1:n, g₂, fmt=:png, ratio=1,
						title="Multigrid Solution Error - Heatmap")
	sfc₂ 	= surface(g₂, fmt=:png, ratio=1,
						legend=false, title="Multigrid Solution Error - Surface")

	plot(htmp₁, sfc₁, htmp₂, sfc₂, layout=(2,2), size=(1000,1000))
end

# ╔═╡ d2b8e000-5772-11eb-3064-ff760324a0fc
norm(a₁, 2) - norm(a₂, 2)

# ╔═╡ 4f1847fe-5777-11eb-29ab-2b345eab322e
@elapsed multigrid(A, b, u, 5, ω, 1e-30, 4)

# ╔═╡ 577a48c0-5777-11eb-2e96-9f131106879a
@elapsed JOR(A(n), b, ω, u, 1e-30, 500)

# ╔═╡ Cell order:
# ╟─2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
# ╠═861aac40-552b-11eb-1151-6b74f5ed3de5
# ╠═9c806a14-551d-11eb-0676-5942da7c95e7
# ╟─c33cb002-553d-11eb-31d3-3176668c06df
# ╟─c68281e6-5539-11eb-10c0-fd665be52ae0
# ╟─34f711ea-554f-11eb-342f-0358c848c965
# ╟─207fd796-5550-11eb-2e66-a93e8f66e338
# ╟─ba523a0e-554f-11eb-162f-8b6e6434d21c
# ╟─4f335f12-573b-11eb-042c-a97c20dd929d
# ╟─8a710356-573b-11eb-11b0-1b418859bb13
# ╠═be97faee-5532-11eb-1466-15d0f84888cf
# ╟─d3ecaa6c-55ef-11eb-032c-632239a2f100
# ╠═5c93118a-5555-11eb-14b1-872b186d3b0b
# ╠═bbfb0164-5771-11eb-09c0-8d8bbf8e7434
# ╠═aafcba2a-5750-11eb-19e9-8fa5b1952b54
# ╠═02c83b80-5546-11eb-2fd1-09fc47ed0d66
# ╠═40bdad12-55fa-11eb-0b00-c57014be1cdf
# ╠═846b2244-56ac-11eb-3b62-e9b1b45d3582
# ╠═c487b42c-55ea-11eb-0af0-2d7b1bcced1f
# ╠═d2b8e000-5772-11eb-3064-ff760324a0fc
# ╠═4f1847fe-5777-11eb-29ab-2b345eab322e
# ╠═577a48c0-5777-11eb-2e96-9f131106879a
