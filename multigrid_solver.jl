### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 861aac40-552b-11eb-1151-6b74f5ed3de5
# Used Packages Index
begin
	using LinearAlgebra 	# Linear Algebra
	using SparseArrays 		# Sparse Arrays Optimization
	using Plots 			# Plotting & Visualization
end ;

# ╔═╡ 2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
md"""
## Notebook: Geometric multigrid for 2D-PDE's
"""

# ╔═╡ c33cb002-553d-11eb-31d3-3176668c06df
md"""
**Constructing the Laplacian 2D-operator and the Problems PDEs**
"""

# ╔═╡ c68281e6-5539-11eb-10c0-fd665be52ae0
html"""
<p style="text-align: center;">
$A u = \Delta u$ = $\frac{∂²u}{∂x²}\ e_x$ + $\frac{∂²u}{∂y²}\ e_y$
</p>
"""

# ╔═╡ 0e90ca56-5849-11eb-15b6-7f4419690f96
md"""
To construct the laplacian operator, we're using the **Kronecker product** with the identity matrix to develop the axe-wise differentiation operators and sum them up lately to obtain Δ.
"""

# ╔═╡ 3bccede2-5849-11eb-21e0-37c579ac744e
html"""
<center>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/cc2eec0b97a4fae13cb04ca7e06687bca1e2c120" alt="Kronecker Product">
<br/>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/88948d4780e5d5fcb6e786d9d4c172ea78ceaabb" alt="Kronecker Product - explicit">
</center>
"""

# ╔═╡ 55b1fd86-584d-11eb-26f3-a739ab86bc4d
md"""Regarding the Kronecker product, it is actually a **computationally expensive** operation, that is why we have to use **sparse matrices** for this extent. The use of sparse matrices is further supported by the sparsity factor of our Laplacian operator that isn't to be ignored.
"""

# ╔═╡ 39a20744-584d-11eb-3376-614cc8c10fde
html"""
<center><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQd4MNLEj5e4xld1062xPxR-xPJ13Q7QXVQSg&usqp=CAU" alt="Operator A for Au = f"></center>
"""

# ╔═╡ ddb578ba-5858-11eb-3775-d3091ced2035
# Dense Array representation taking all 4x5 memory spaces
a = [3 0 0 0 0; 0 0 2 9 0; 0 0 0 0 1; 0 0 0 4 0]

# ╔═╡ 678e0c28-5859-11eb-0bd5-b935bd90eed0
# Number of Non-zeros of a
nnz(sparse(a))

# ╔═╡ f89b45ce-5858-11eb-2ae9-07c0a3e86908
# Sparse Array variant taking only nnz(a) memory spaces
sparse(a)

# ╔═╡ 34f711ea-554f-11eb-342f-0358c848c965
md"""We then need to transform our differential equation into a linear problem in the form of $\textbf{Au=f}$. For that, we only have to use the matrix $σI$ to then factorize the equation by $u(x,y)$.
"""

# ╔═╡ ba523a0e-554f-11eb-162f-8b6e6434d21c
html"""
<center>
$-Δ u(x,y) + σu(x,y) = f(x,y)\ \ \ \ (P_1)$<br/>
$(σI-Δ)u(x,y) = f(x,y)$<br/>
$A_1=σI-Δ$
</center>
"""

# ╔═╡ 4f335f12-573b-11eb-042c-a97c20dd929d
md"""
As for the Anisotropic problem $P_2$, the construction is similar but only using the partial derivatives
"""

# ╔═╡ 8a710356-573b-11eb-11b0-1b418859bb13
html"""
<center>
$-\frac{\partial^2 u(x,y)}{∂x²} - \epsilon \frac{\partial^2 u(x,y)}{∂y²} = f(x,y)\ \ \ \ (P_2)$<br/>
$A_2=(\frac{\partial^2}{∂x²} - \epsilon \frac{\partial^2}{∂y²})I$
</center>
"""

# ╔═╡ be97faee-5532-11eb-1466-15d0f84888cf
begin
	"""
		A₁(n::Int, σ::Float64)

	A function that returns the Poisson probem's linear operator.
	"""
	function A₁(n::Int, σ::Float64)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (n^2) * kron(sparse(∂²), I(n))	# kron is the Kronecker product
		∂y² = (n^2) * kron(I(n), sparse(∂²))	# h is the unit of displacement
		Δ 	= ∂x² + ∂y²
		return σ * I(n^2) - Δ
	end
	
	"""
		A₁(σ::Float64)

	A function that returns the Poisson probem's linear operator.
	This variant takes the σ value as input to construct an A₁ operator constructor that only takes the differentiable matrix dimension as input.
	"""
	function A₁(σ::Float64)
		return n -> A₁(n, σ)
	end

	"""
		A₂(n::Int, ϵ::Float64)

	A function that returns the Anisotropic probem's linear operator.
	"""
	function A₂(n)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (n^2) * kron(sparse(∂²), I(n))	# kron is the Kronecker product
		∂y² = (n^2) * kron(I(n), sparse(∂²))	# h is the unit of displacement
		return - ∂x² - ϵ * ∂y²
	end
	
	"""
		A₂(ϵ::Float64)

	A function that returns the Poisson probem's linear operator.
	This variant takes the ϵ value as input to construct an A₂ operator constructor that only takes the differentiable matrix dimension as input.
	"""
	function A₂(ϵ::Float64)
		return n -> A₁(n, ϵ)
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

# ╔═╡ e2f0fef8-584e-11eb-2f0c-49b4cb4009ba
md"""For the stationary smoothers, we're each time choosing one of the defined functions below either to be used on its own for comparison or to use it as the smoother in our Multigrid implementation.
"""

# ╔═╡ bbfb0164-5771-11eb-09c0-8d8bbf8e7434
"""
    Jacobi(A, b, u₀, ϵ, maxiter)

Jacobi iterative smoother.
"""
function Jacobi(A, b, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10)
    u = u₀; n = Int(sqrt(size(A, 1))); iter = 0
	M = Diagonal(A)
	N = UnitLowerTriangular(A) + UnitUpperTriangular(A) - 2*I(n^2)
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
		(norm(r, 2) > ϵ) || break 			# Convergence check
	end
	return u
end ;

# ╔═╡ 25766a80-583e-11eb-1286-4945ee2b6fdb
"""
    SOR(A, b, ω, u₀, ϵ, maxiter)

Successive Over Relaxation iterative smoother - Gauss Seidel variant.
"""
function SOR(A, b, ω, u₀ = zeros(size(A, 1)), ϵ = 1e-7, maxiter = 10)
    u = u₀; n = Int(sqrt(size(A, 1))); iter = 0
	D = Diagonal(A)
	L = UnitLowerTriangular(A) - I(n^2)
	U = UnitUpperTriangular(A) - I(n^2)
	while iter <= maxiter
		iter += 1
		u = inv(D + ω * L) * (ω * b - (ω * U + (ω-1) * D) * u)
		(norm(b - A*u, 2) > ϵ) || break 	# Convergence check
	end
	return u
end ;

# ╔═╡ 27af4156-5851-11eb-36ba-3f11e34bee5e
html"""
<center><img width="75%" src="https://www.researchgate.net/profile/Colin_Fox/publication/231064411/figure/fig1/AS:408425306574849@1474387589707/Schematic-of-the-V-cycle-multigrid-iterative-algorithm.png" alt="Multigrid V-cycle"></center>
"""

# ╔═╡ a874e446-5851-11eb-0766-17b85ae44f46
md"""As shown in the simple form of grid-spacing in a **V-cycle** multigrid, we need some **interpolation** operators to be used to move from one grid to another:
- *Restriction*: From a *fine* grid to a *coarser* one
- *Prolongation*: From a *coarse* grid to a *finer* one

Therefore, the methods defined below will be used interchangeably.

Also, as we're treating 2 different problems:
- *isotropic*: $A_1u(x,y)=f(x,y)$
- *$\epsilon$-anisotropic*: $A_2u(x,y)=f(x,y)$

That's why we're going to need two different strategies for grids interpolation as we're not moving in the same space for both problems.
"""

# ╔═╡ 40bdad12-55fa-11eb-0b00-c57014be1cdf
# Multigrid's **Isotropic** Interpolation operators
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
			g[i, j] = 0.5*grid[i÷2, j÷2]
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

# ╔═╡ ab3263ae-581b-11eb-022f-217f547efa3a
html"""
<center><img src="https://ars.els-cdn.com/content/image/1-s2.0-S0377042716300577-fx1.jpg" alt="V-cycle Multigrid Iteration"></center>
"""

# ╔═╡ 0a6287c6-5819-11eb-2b9c-ebe99da1e82f
"""
    multigrid(A, b, u, l, ω, ϵ, steps,
			  restrict, prolong, iter)

l-level Multigrid cycle. Each sub-level dive is held with x10 smoothing iterations.
	- A: the PDE's linear operator constructor
	- restrict: function for grid restriction
	- prolong: function for grid prolongation
	- iter: number of iterations of **current** grid smoothing
	- steps: number of multigrid callbacks per cycle:
		- 1 (default) is for V-cycle
		- 2 is for normal W-cycle
		- n (> 2) is for W-cycles with n sub-refinements
"""
function multigrid(A, b, u, l, ω, ϵ=1e-7, steps=1,
		restrict=injection, prolong=enlarge, iter=10)
	n  = Int(sqrt(size(b, 1)))
	Aₙ = A(n)
	if l == 0
		# We can use a direct solver instead
		# u = Array(Aₙ) \ b
		u = JOR(Aₙ, b, ω, u, ϵ, iter)
	else
		u = JOR(Aₙ, b, ω, u, ϵ, iter)			# Pre-smoothing

		# Defect restriction
		r = reshape(b - Aₙ*u, (n, n)) |> transpose |>
			restrict |> transpose |> vec

		# Coarse-level Correction
		δᵣ = zeros(size(r))
		for i=1:steps
			δᵣ = multigrid(A, r, δᵣ, l-1, ω, ϵ,
					steps, restrict, prolong, iter*3)
		end

		# Defect Prolongation δᵣ → δ
		δ = reshape(δᵣ, (n÷2, n÷2)) |> transpose |>
			prolong |> transpose |> vec
		
		u += δ 									# Correction
		u = JOR(Aₙ, b, ω, u, ϵ, iter)			# Post-smoothing
	end
	return u
end ;

# ╔═╡ 628964aa-5854-11eb-2d40-45bc01ae2357
md"""
**Simulation and Results Interpretation**

We start by running our *isotropic* problem resolution governed by the $A_1$ operator. We're also only interested in the *unit square* resolution of the function $f$, and for it we discretize the space on $n=1024$ units, $h=1/n$ being the displacement unit.
"""

# ╔═╡ cdfd18c0-57fd-11eb-2dbb-415d4e27fa5d
begin
	n = 1024
	h = 1 / n
	ω = 0.95
	σ = 0.7
	A = A₁(σ)
	u = zeros(n^2)
	f = 3
	b = [sin(2π*f*i*j) for i=0:h:1-h for j=0:h:1-h]
end ;

# ╔═╡ 5c49ed5e-585a-11eb-32d2-0ddecc88d963
# Jacobi Over Relaxation
@elapsed u₁ = JOR(A(n), b, ω, u, 1e-30, 50)

# ╔═╡ 7d83e592-585a-11eb-2a5d-1f8c8fb2fd8b
# Multigrid V-cycle
@elapsed u₂ = multigrid(A, b, u, 3, ω, 1e-30, 1)

# ╔═╡ 046db82e-585d-11eb-158b-c760258806ab
md"""
Examining the cells above, we can notice that both solvers took a similar time to finish their iterations. If we then thoroughly examine the written multigrid code, we'll find out that for the current simulation, while it yielded nearly the **same execution-time** as the Jacobi-Over-Relaxation method, we're actually running a *3-level V-cycle multigrid* that computes over **800 iterations** and that also have given a by-far **better approximation** than its iterative counterpart even when running most of those iterations on the *coarser* grid 128x128 instead of the original 1024x1024 one.
"""

# ╔═╡ 6ce6509e-5783-11eb-1066-515dfad96fec
# Comparing solutions error rates
begin
	eⱼ 		= b - A(n)*u₁ 			# Jacobi solution error
	eₘ 		= b - A(n)*u₂ 			# Multigrid solution error
	fastr 	= norm(eₘ, 2) - norm(eⱼ, 2)
	@show "||eₘ|| $(fastr < 0 ? '<' : '>') ||eⱼ||"
end

# ╔═╡ b69d80da-5856-11eb-1fae-5fdf08297723
@show "Error norm gain: $(round(fastr, digits=5))"

# ╔═╡ 421e9880-585f-11eb-1a4a-db98dbf494d6
md"""
The near-similarity of the execution-time between both solvers is actually due to the fact that the 300 iterations ran by the Multigrid are equivalent to only 48 ones of the Jacobi-Over-Relaxation method.
"""

# ╔═╡ 3a7b6bbe-5853-11eb-12d0-2d3907b68685
@show "Multigrid JOR-relative cost: $(20+(20*3)/4+(20*9)/16+(20*27)/256) iterations"

# ╔═╡ 441ed93e-5860-11eb-07f0-f70c9720c8b5
md"""**Solutions Visualization & interpretation**"""

# ╔═╡ 0298049a-57b4-11eb-0cb2-e3524c90e250
# Visualizing solution errors for the 2D-Poisson Problem
begin
	# Jacobi-OR
	g₁ 		= reshape(eⱼ, (n, n))' |> Array
	htmp₁ 	= heatmap(1:n, 1:n, g₁, fmt=:png, ratio=1,
						title="JOR Solution Error")

	# Multigrid V-cycle
	g₂ 		= reshape(eₘ, (n, n))' |> Array
	htmp₂ 	= heatmap(1:n, 1:n, g₂, fmt=:png, ratio=1,
						title="Multigrid Solution Error")

	plot(htmp₁, htmp₂, layout=(1,2), size=(1000,500))
end

# ╔═╡ 870b1224-5860-11eb-3d77-1d6a0bc7f8a6
md"""From the graphs shown above and below, we can tell that even when both obtained solutions seem to be somewhat similar *(Multigrid's being dimmer which by the graph's legend means a **nearer to 0 error**)*, the solutions difference distribution tell us that the iterative method had a bit of a struggle polishing the *lowest frequencies* in only 50 iterations.
"""

# ╔═╡ 71deb1c0-5843-11eb-315c-b9cee350e560
# Visualizing solutions difference for the 2D-Poisson Problem
begin
	htmp 	= heatmap(1:n, 1:n, g₂ - g₁, fmt=:png, ratio=1,
						title="Solutions Difference Heatmap")
	sfc 	= surface(g₂ - g₁, fmt=:png, ratio=1, legend=false,
						title="Solutions Difference Distribution")
	plot(htmp, sfc, layout=(1,2), size=(1000,500))
end

# ╔═╡ 32f264d2-584a-11eb-01ba-3b4ea67003df
md"""**V-cycle vs W-cycle**

We'll now proceed to compare the convergence of the V-cycle and the W-cycle variants of the multigrid solver for various $\sigma$ and $n$ levels for the problem operator $A₁(\sigma, n)$."""

# ╔═╡ e70c7750-5863-11eb-38ea-157cc0b19d9b
begin
	σ₁ 	= 0.7
	n₁ 	= 1024
	h₁ 	= 1 / n₁
	uₙ₁ = zeros(n₁^2)
	bₙ₁ = [sin(2π*f*i*j) for i=0:h₁:1-h₁ for j=0:h₁:1-h₁]
	A₁₁ = A₁(σ₁)
end ;

# ╔═╡ 35ce219c-5869-11eb-39f9-577d4de271b1
# Multigrid V-cycle
@elapsed uᵢ₁ = multigrid(A₁₁, bₙ₁, uₙ₁, 3, ω, 1e-30, 1)

# ╔═╡ 5757f624-5869-11eb-3006-6fd9f81ba58c
# Multigrid 2-steps W-cycle
@elapsed uⱼ₁ = multigrid(A₁₁, bₙ₁, uₙ₁, 3, ω, 1e-30, 2)

# ╔═╡ ebf9da62-586a-11eb-0386-7576e50ef520
# Comparing solutions error rates
begin
	eᵢ₁ 	= bₙ₁ - A₁₁(n)*uᵢ₁ 			# V-cycle solution error
	eⱼ₁ 	= bₙ₁ - A₁₁(n)*uⱼ₁ 			# W-cycle solution error
	fastr₁ 	= norm(eᵢ₁, 2) - norm(eⱼ₁, 2)
	@show "||eᵢ₁|| $(fastr₁ < 0 ? '<' : '>') ||eⱼ₁||"
end

# ╔═╡ 4a51a37e-586b-11eb-2304-a73fbb512158
@show "Error norm gain: $(round(fastr₁, digits=5))"

# ╔═╡ 69ca47ba-586b-11eb-18a6-8128dd4670ce
# Visualizing solution errors
begin
	# Jacobi-OR
	gᵢ₁ 	= reshape(eᵢ₁, (n₁, n₁))' |> Array
	htmpᵢ₁ 	= heatmap(1:n₁, 1:n₁, gᵢ₁, fmt=:png, ratio=1,
						title="V-cycle - n = 1024 - σ = 0.7")

	# Multigrid V-cycle
	gⱼ₁ 	= reshape(eⱼ₁, (n₁, n₁))' |> Array
	htmpⱼ₁ 	= heatmap(1:n₁, 1:n₁, gⱼ₁, fmt=:png, ratio=1,
						title="W-cycle - n = 1024 - σ = 0.7")

	plot(htmpᵢ₁, htmpⱼ₁, layout=(1,2), size=(1000,500))
end

# ╔═╡ 5f94664a-586b-11eb-0ed7-c92bb9aa59e0


# ╔═╡ 446a6f40-586b-11eb-282a-f94ee14269a2


# ╔═╡ edf8d17e-586a-11eb-2032-4743e1a4ccaf


# ╔═╡ ee1653e0-586a-11eb-132c-01f49c5a1fa9


# ╔═╡ ee2a8d9c-586a-11eb-3ae0-47158f69bbc0


# ╔═╡ dac43802-586a-11eb-3b90-8fbf23bfb42b


# ╔═╡ Cell order:
# ╟─2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
# ╠═861aac40-552b-11eb-1151-6b74f5ed3de5
# ╟─c33cb002-553d-11eb-31d3-3176668c06df
# ╟─c68281e6-5539-11eb-10c0-fd665be52ae0
# ╟─0e90ca56-5849-11eb-15b6-7f4419690f96
# ╟─3bccede2-5849-11eb-21e0-37c579ac744e
# ╟─55b1fd86-584d-11eb-26f3-a739ab86bc4d
# ╟─39a20744-584d-11eb-3376-614cc8c10fde
# ╠═ddb578ba-5858-11eb-3775-d3091ced2035
# ╠═678e0c28-5859-11eb-0bd5-b935bd90eed0
# ╠═f89b45ce-5858-11eb-2ae9-07c0a3e86908
# ╟─34f711ea-554f-11eb-342f-0358c848c965
# ╟─ba523a0e-554f-11eb-162f-8b6e6434d21c
# ╟─4f335f12-573b-11eb-042c-a97c20dd929d
# ╟─8a710356-573b-11eb-11b0-1b418859bb13
# ╠═be97faee-5532-11eb-1466-15d0f84888cf
# ╟─d3ecaa6c-55ef-11eb-032c-632239a2f100
# ╠═5c93118a-5555-11eb-14b1-872b186d3b0b
# ╟─e2f0fef8-584e-11eb-2f0c-49b4cb4009ba
# ╠═bbfb0164-5771-11eb-09c0-8d8bbf8e7434
# ╠═aafcba2a-5750-11eb-19e9-8fa5b1952b54
# ╠═25766a80-583e-11eb-1286-4945ee2b6fdb
# ╟─27af4156-5851-11eb-36ba-3f11e34bee5e
# ╟─a874e446-5851-11eb-0766-17b85ae44f46
# ╠═40bdad12-55fa-11eb-0b00-c57014be1cdf
# ╟─ab3263ae-581b-11eb-022f-217f547efa3a
# ╠═0a6287c6-5819-11eb-2b9c-ebe99da1e82f
# ╟─628964aa-5854-11eb-2d40-45bc01ae2357
# ╠═cdfd18c0-57fd-11eb-2dbb-415d4e27fa5d
# ╠═5c49ed5e-585a-11eb-32d2-0ddecc88d963
# ╠═7d83e592-585a-11eb-2a5d-1f8c8fb2fd8b
# ╟─046db82e-585d-11eb-158b-c760258806ab
# ╠═6ce6509e-5783-11eb-1066-515dfad96fec
# ╟─b69d80da-5856-11eb-1fae-5fdf08297723
# ╟─421e9880-585f-11eb-1a4a-db98dbf494d6
# ╟─3a7b6bbe-5853-11eb-12d0-2d3907b68685
# ╟─441ed93e-5860-11eb-07f0-f70c9720c8b5
# ╠═0298049a-57b4-11eb-0cb2-e3524c90e250
# ╟─870b1224-5860-11eb-3d77-1d6a0bc7f8a6
# ╠═71deb1c0-5843-11eb-315c-b9cee350e560
# ╟─32f264d2-584a-11eb-01ba-3b4ea67003df
# ╠═e70c7750-5863-11eb-38ea-157cc0b19d9b
# ╠═35ce219c-5869-11eb-39f9-577d4de271b1
# ╠═5757f624-5869-11eb-3006-6fd9f81ba58c
# ╠═ebf9da62-586a-11eb-0386-7576e50ef520
# ╟─4a51a37e-586b-11eb-2304-a73fbb512158
# ╠═69ca47ba-586b-11eb-18a6-8128dd4670ce
# ╠═5f94664a-586b-11eb-0ed7-c92bb9aa59e0
# ╠═446a6f40-586b-11eb-282a-f94ee14269a2
# ╠═edf8d17e-586a-11eb-2032-4743e1a4ccaf
# ╠═ee1653e0-586a-11eb-132c-01f49c5a1fa9
# ╠═ee2a8d9c-586a-11eb-3ae0-47158f69bbc0
# ╠═dac43802-586a-11eb-3b90-8fbf23bfb42b
