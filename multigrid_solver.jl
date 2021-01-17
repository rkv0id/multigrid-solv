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

# ╔═╡ f893039e-58be-11eb-0c48-f9f499f6856d
md"""
Solving Partial Differential Equations has never been an easy task on mathematicians. In fact, in many cases it's even considered *impossible* to do with our current mathematics, so our best way to approach it is with numerical/computational approximations. It's this exact field that has specialized in solving PDE's numerically: *Using Iterative methods, Fourrier Transforms, Finite Differences, etc..*

This project is about the **Multigrid methods** which are one of the most interesting assets of the numerical computing toolbox. We're also going to do some benchmarking of their performance regarding their iterative counterparts.

Finally, to be able to start, we should first do the initial step of the numerical problem resolution framework: **Discretizing the PDE space**
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
<img width="40%" src="https://i.stack.imgur.com/GYj20.png">
<br/>
<figure>
<img src="data:image/png;base64,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" alt="Operator A for Au = f">
<figcaption>$\Delta$ Operator Matrix</figcaption>
</figure>
</center>
"""

# ╔═╡ 55b1fd86-584d-11eb-26f3-a739ab86bc4d
md"""Regarding the Kronecker product, it is actually a **computationally expensive** operation, that is why we have to use **sparse matrices** for this extent. The use of sparse matrices is further supported by the sparsity factor of our Laplacian operator that isn't to be ignored.
"""

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
	function A₁(n::Int, σ::Float64)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (n^2) * kron(sparse(∂²), I(n))	# kron is the Kronecker product
		∂y² = (n^2) * kron(I(n), sparse(∂²))	# h is the unit of displacement
		Δ 	= ∂x² + ∂y²
		return σ * I(n^2) - Δ
	end
	
	function A₂(n::Int, ϵ::Float64)
		∂² 	= Tridiagonal(ones(n-1), -2 * ones(n), ones(n-1))
		∂x² = (n^2) * kron(sparse(∂²), I(n))	# kron is the Kronecker product
		∂y² = (n^2) * kron(I(n), sparse(∂²))	# h is the unit of displacement
		return - ∂x² - ϵ * ∂y²
	end
	
	A₁(σ::Float64) = n -> A₁(n, σ)
	A₂(ϵ::Float64) = n -> A₂(n, ϵ)
end ;

# ╔═╡ 011d1544-58de-11eb-31bc-fb0262856677
md"""
Before jumping to start modelling the solvers, we need first to iterate on the boundaries conditions. The PDE's we're solving are both using the same boundaries conditions $u(x,y) = 0$ on $∂Ω$ so we're just coding them in a function that will be called after each solver iteration to reinject the boundaries so that we don't miss the track of them.
"""

# ╔═╡ 3f38303e-58de-11eb-3bb5-ef7da4dfcce3
function boundaries(v)
	n = Int(sqrt(size(v,1)))
	gridᵥ = reshape(v, (n, n))'
	gridᵥ[1,:] .= 0
	gridᵥ[end,:] .= 0
	gridᵥ[:,1] .= 0
	gridᵥ[:,end] .= 0
	return gridᵥ |> transpose |> vec
end ;

# ╔═╡ e2f0fef8-584e-11eb-2f0c-49b4cb4009ba
md"""
**Constructing the solvers algorithms**

For the stationary smoothers, we're each time choosing one of the defined functions below either to be used on its own for comparison or to use it as the smoother in our Multigrid implementation.
"""

# ╔═╡ bbfb0164-5771-11eb-09c0-8d8bbf8e7434
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
end ;

# ╔═╡ aafcba2a-5750-11eb-19e9-8fa5b1952b54
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
end ;

# ╔═╡ 25766a80-583e-11eb-1286-4945ee2b6fdb
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
@elapsed u₁ = JOR(A(n), b, ω, u, 1e-30, 50, true)

# ╔═╡ 7d83e592-585a-11eb-2a5d-1f8c8fb2fd8b
# Multigrid V-cycle
@elapsed u₂ = multigrid(A, b, u, 3, ω, 1e-30, 1, injection, linearize)

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
@show "Error norm gain: $(round(abs(fastr), digits=5))"

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
md"""From the graphs shown above and below, we can tell that the multigrid's solution is by far a better solution than the iterative JOR solver *(Multigrid's being dimmer and softer which by the graph's legend means a **nearer to 0 error**)*. Also, the solutions difference distribution tells us even more about the struggle that Jacobi-Over-Relaxation method had trying to polish the *lowest frequencies* in only 50 iterations.
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
"""

# ╔═╡ d27ae6f8-586d-11eb-1ed3-1da566271a73
html"""
<center>
<img width="75%" src="https://d3i71xaburhd42.cloudfront.net/5dd2565fee1cdda8cf9ba444262f3ba3b5323219/73-Figure4.8-1.png"/>
</center>
"""

# ╔═╡ d25069b4-5874-11eb-3876-dd2e4b0cce2a
md"""
We'll now proceed to compare the convergence of the V-cycle and the W-cycle variants of the multigrid solver for various $\sigma$ and $n$ levels for the problem operator $A₁(\sigma, n)$."""

# ╔═╡ b1ee0492-5874-11eb-35ef-27791afce7a1
md"""**1-** $n = 1024\ ,\ \sigma = 0.7$"""

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

# ╔═╡ 624f194a-5873-11eb-1ed4-112551856dfe
md"""
By just executing both solvers, we can already notice the difference in time between both solvers. The difference shown above actually makes sense: Our W-cycle is a **2-step** one on a **3-level** multigrid, which means it takes the exact form of a W, then we're executing the same number of iterations but **twice**. This gets reflected by the *more-than-twice* execution time we have with the W-cycle, compared to the V-cycle.
"""

# ╔═╡ ebf9da62-586a-11eb-0386-7576e50ef520
# Comparing solutions error rates
begin
	eᵢ₁ 	= bₙ₁ - A₁₁(n)*uᵢ₁ 			# V-cycle solution error
	eⱼ₁ 	= bₙ₁ - A₁₁(n)*uⱼ₁ 			# W-cycle solution error
	fastr₁ 	= norm(eᵢ₁, 2) - norm(eⱼ₁, 2)
	@show "||eᵥ|| $(fastr₁ < 0 ? '<' : '>') ||eᵪ||"
end

# ╔═╡ 04354736-5877-11eb-0a11-dd87cbdf782e
md"""From here, we can confirm that the W-cycle was actually worth the wait, as it did help smoothing out the error in a more extreme way than the V-cycle did.
"""

# ╔═╡ 4a51a37e-586b-11eb-2304-a73fbb512158
@show "Error norm gain: $(round(fastr₁, digits=5))"

# ╔═╡ 69ca47ba-586b-11eb-18a6-8128dd4670ce
# Visualizing solution errors
begin
	# V-cycle
	gᵢ₁ 	= reshape(eᵢ₁, (n₁, n₁))' |> Array
	htmpᵢ₁ 	= heatmap(1:n₁, 1:n₁, gᵢ₁, fmt=:png, ratio=1,
						title="V-cycle - n = 1024 - σ = 0.7")

	# W-cycle
	gⱼ₁ 	= reshape(eⱼ₁, (n₁, n₁))' |> Array
	htmpⱼ₁ 	= heatmap(1:n₁, 1:n₁, gⱼ₁, fmt=:png, ratio=1,
						title="W-cycle - n = 1024 - σ = 0.7")

	plot(htmpᵢ₁, htmpⱼ₁, layout=(1,2), size=(1000,500))
end

# ╔═╡ 3944a45c-5873-11eb-1af2-65ad1dbc48df
md"""**2-** $n = 1024\ ,\ \sigma = 7$"""

# ╔═╡ c4d86ad0-586e-11eb-3296-8f61deff1e47
begin
	σ₂ 	= 7.
	A₁₂ = A₁(σ₂)
	uᵢ₂ = multigrid(A₁₂, bₙ₁, uₙ₁, 3, ω, 1e-30, 1)		# Multigrid V-cycle
	uⱼ₂ = multigrid(A₁₂, bₙ₁, uₙ₁, 3, ω, 1e-30, 2)		# Multigrid W-cycle
	eᵢ₂ 	= bₙ₁ - A₁₂(n)*uᵢ₂ 							# V-cycle solution error
	eⱼ₂ 	= bₙ₁ - A₁₂(n)*uⱼ₂ 							# W-cycle solution error
	fastr₂ 	= norm(eᵢ₂, 2) - norm(eⱼ₂, 2)
end ;

# ╔═╡ 1037e1a6-5872-11eb-0820-bd41a05d1af1
@show "||eᵥ|| $(fastr₂ < 0 ? '<' : '>') ||eᵪ||"

# ╔═╡ f0e013b0-586e-11eb-2457-f917877375cb
@show "Error norm gain: $(round(abs(fastr₂), digits=5))"

# ╔═╡ 18c95184-5871-11eb-0c6e-d3e710f51be2
@show "Error norm gain - V-cycle: $(round(abs(norm(eᵢ₂, 2) - norm(eᵢ₁, 2)), digits=5))"

# ╔═╡ 362bda12-5871-11eb-3c16-8d53b233a01d
@show "Error norm gain - W-cycle: $(round(abs(norm(eⱼ₂, 2) - norm(eⱼ₁, 2)), digits=5))"

# ╔═╡ 41c925c2-5877-11eb-2e1d-af5b5f0cd74a
md"""
Increasing $\sigma$ by a factor of x10 lead both solvers to better solutions relatively, but it did have the same effect on both of them as the error gain of the W-cycle over the V-cycle is still nearly the same.
"""

# ╔═╡ 3a56d20a-5875-11eb-03c0-a5b112d19efe
md"""**3-** $n = 1024\ ,\ \sigma = 70$"""

# ╔═╡ 3a5cb224-5875-11eb-2c1b-97436009b686
begin
	σ₃ 	= 70.
	A₁₃ = A₁(σ₃)
	uᵢ₃ = multigrid(A₁₃, bₙ₁, uₙ₁, 3, ω, 1e-30, 1)		# Multigrid V-cycle
	uⱼ₃ = multigrid(A₁₃, bₙ₁, uₙ₁, 3, ω, 1e-30, 2)		# Multigrid W-cycle
	eᵢ₃ 	= bₙ₁ - A₁₃(n)*uᵢ₃ 							# V-cycle solution error
	eⱼ₃ 	= bₙ₁ - A₁₃(n)*uⱼ₃ 							# W-cycle solution error
	fastr₃ 	= norm(eᵢ₃, 2) - norm(eⱼ₃, 2)
end ;

# ╔═╡ 3a6569f0-5875-11eb-344d-5bf556e0aa8d
@show "||eᵥ|| $(fastr₃ < 0 ? '<' : '>') ||eᵪ||"

# ╔═╡ 3a668812-5875-11eb-326f-87f50db349c8
@show "Error norm gain: $(round(abs(fastr₃), digits=5))"

# ╔═╡ 3a784ba6-5875-11eb-1979-7d9a299ea243
@show "Error norm gain - V-cycle: $(round(abs(norm(eᵢ₃, 2) - norm(eᵢ₂, 2)), digits=5))"

# ╔═╡ 3a805846-5875-11eb-0f01-071787fe43fc
@show "Error norm gain - W-cycle: $(round(abs(norm(eⱼ₃, 2) - norm(eⱼ₂, 2)), digits=5))"

# ╔═╡ 446a6f40-586b-11eb-282a-f94ee14269a2
md"""
Increasing $\sigma$ by a factor of x100 lead both solvers to better solutions, and this time, it started showing its effect on the W side of things more than on the V-cycle's side. But still, it also increased the error-wise gain of the V-cycle multigrid solver too.
"""

# ╔═╡ 1832d306-5878-11eb-311c-11fabb6bbb2c
md"""**4-** $n = 256\ ,\ \sigma = 7$"""

# ╔═╡ 18391a04-5878-11eb-3287-57163da7c4f8
begin
	n₂ 	= 256
	h₂ 	= 1 / n₂
	uₙ₂ = zeros(n₂^2)
	bₙ₂ = [sin(2π*f*i*j) for i=0:h₂:1-h₂ for j=0:h₂:1-h₂]
	uᵢ₄ = multigrid(A₁₂, bₙ₂, uₙ₂, 3, ω, 1e-30, 1)		# Multigrid V-cycle
	uⱼ₄ = multigrid(A₁₂, bₙ₂, uₙ₂, 3, ω, 1e-30, 2)		# Multigrid W-cycle
	eᵢ₄ 	= bₙ₂ - A₁₂(n₂)*uᵢ₄ 						# V-cycle solution error
	eⱼ₄ 	= bₙ₂ - A₁₂(n₂)*uⱼ₄ 						# W-cycle solution error
	fastr₄ 	= norm(eᵢ₄, 2) - norm(eⱼ₄, 2)
end ;

# ╔═╡ 183e589a-5878-11eb-3584-b37c8b96fb45
@show "||eᵥ|| $(fastr₄ < 0 ? '<' : '>') ||eᵪ||"

# ╔═╡ 18484de4-5878-11eb-22f5-f37516a22b29
@show "Error norm gain: $(round(abs(fastr₄), digits=5))"

# ╔═╡ 184a1dc4-5878-11eb-3513-9fe97c35da4c
@show "Error norm gain - V-cycle: $(round(abs(norm(eᵢ₄, 2) - norm(eᵢ₂, 2)), digits=5))"

# ╔═╡ 1854a65c-5878-11eb-2473-b7acbd68e216
@show "Error norm gain - W-cycle: $(round(abs(norm(eⱼ₄, 2) - norm(eⱼ₂, 2)), digits=5))"

# ╔═╡ 1855c320-5878-11eb-0468-b76ec811f012
md"""
Decreasing $n$ by a factor of x4 lead both solvers as usual to **better solutions**, this time, having a **huge effect on the V-cycle**'s side compared to the W-cycle as the difference between the solutions is converging to none. Still, the latter is the better solver on these simulation settings.

As for the effect this had on improving the approximation, this was expected: Actually, as we go down in the grid's size, we're releasing the multigrid from the burden of having to interpolate on bigger grids *(the ones that we got rid of when we divided the fine size by 4)* which means the solver now is **less error-prone** than on the bigger grid case. Also, let's not forget that by doing such, we're giving the multigrid the opportunity to **spend the same number of iterations but on coarser grids**, this helps it converge faster than before. And this is exactly why we witnessed these huge evolutions on the approximation error.
"""

# ╔═╡ b03feb94-587c-11eb-04b3-33363338be39
md"""**5-** $n = 32\ ,\ \sigma = 7$"""

# ╔═╡ b043fe28-587c-11eb-31a6-d9199b0a3231
begin
	n₃ 	= 32
	h₃ 	= 1 / n₃
	uₙ₃ = zeros(n₃^2)
	bₙ₃ = [sin(2π*f*i*j) for i=0:h₃:1-h₃ for j=0:h₃:1-h₃]
	uᵢ₅ = multigrid(A₁₂, bₙ₃, uₙ₃, 3, ω, 1e-30, 1)		# Multigrid V-cycle
	uⱼ₅ = multigrid(A₁₂, bₙ₃, uₙ₃, 3, ω, 1e-30, 2)		# Multigrid W-cycle
	eᵢ₅ 	= bₙ₃ - A₁₂(n₃)*uᵢ₅ 						# V-cycle solution error
	eⱼ₅ 	= bₙ₃ - A₁₂(n₃)*uⱼ₅ 						# W-cycle solution error
	fastr₅ 	= norm(eᵢ₅, 2) - norm(eⱼ₅, 2)
end ;

# ╔═╡ b045b128-587c-11eb-1064-8166fe875bd9
@show "||eᵥ|| $(fastr₅ < 0 ? '<' : '>') ||eᵪ||"

# ╔═╡ b053d5a2-587c-11eb-0fcf-053648eb60d9
@show "Error norm gain: $(round(abs(fastr₅), digits=5))"

# ╔═╡ b0551278-587c-11eb-174f-cf692bcd0675
@show "Error norm gain - V-cycle: $(round(abs(norm(eᵢ₅, 2) - norm(eᵢ₂, 2)), digits=5))"

# ╔═╡ b060aa6e-587c-11eb-3ac9-5935786e20c1
@show "Error norm gain - W-cycle: $(round(abs(norm(eⱼ₅, 2) - norm(eⱼ₂, 2)), digits=5))"

# ╔═╡ b061db8a-587c-11eb-312a-fd93e66e3039
md"""
On the abstract side of the story, as the theory states, the W-cycle is still a better solver than the V-cycle. But as we're still approaching coarser grid sizes *(here dividing by a factor of x32)*, the fine-tuning is happening on even coarser grids *(with a size of 4x4 in these settings)* so increasing the number of iterations or further smoothing out the approximation n-times *(as it's the case for the W-cycle)* **isn't going to make any difference**.

**Restriction Operators**

Trying to visualize the workings of the Multigrid solver, we always stumble on the matrix transformations we're using to interpolate the grid we have to a coarser one *(restriction)* or to a finer one *(prolongation)*. As we already created these 3 operators *(see code in the solvers implementation part)*, we're going to compare them according to what they give in term of approximation error.

For this benchmarking, we're conducting the $n=1024$ , $\sigma=7$ simulation. We're also going to use the **4-level 2-steps W-cycle** multigrid equipped with the **linearization** prolongation operator for better results and more complexity to be able to well-assess the performance difference between all runs.
"""

# ╔═╡ 5cfbb91a-5881-11eb-39ed-817b986b9832
# Restriction by Injection
@elapsed uⱼ₆ = multigrid(A₁₂, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, injection, linearize)

# ╔═╡ 5d1657a0-5881-11eb-2784-0d213e722e7f
# Restriction by Halfweighting
@elapsed uⱼ₇ = multigrid(A₁₂, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, halfweight, linearize)

# ╔═╡ 5d2bbe10-5881-11eb-3b4c-3984a432b8f1
# Restriction by Fullweighting
@elapsed uⱼ₈ = multigrid(A₁₂, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, fullweight, linearize)

# ╔═╡ 1a70c16a-5886-11eb-188e-317a78df7b25
md"""
From the execution-times of the 3 variants, we can conclude that thanks to Julia's loops optimization, we can use the 3 restriction operators interchangeably without any loss in performance.
"""

# ╔═╡ 5d41df08-5881-11eb-3abc-6f4480396028
# Comparing solutions error rates
begin
	eⱼ₆ 	= bₙ₁ - A₁₂(n)*uⱼ₆ 			# Injection solution error - eᵢ
	eⱼ₇ 	= bₙ₁ - A₁₂(n)*uⱼ₇ 			# Halfweighting solution error - eₕ
	eⱼ₈ 	= bₙ₁ - A₁₂(n)*uⱼ₈ 			# Fullweighting solution error - eᵩ
	fastr₆₇ = norm(eⱼ₆, 2) - norm(eⱼ₇, 2)
	fastr₆₈ = norm(eⱼ₆, 2) - norm(eⱼ₈, 2)
	fastr₇₈ = norm(eⱼ₇, 2) - norm(eⱼ₈, 2)
end ;

# ╔═╡ 5afc977c-5881-11eb-3629-ef6b8f815404
@show "||eₕ|| $(fastr₇₈ < 0 ? '<' : '>') ||eᵩ||"

# ╔═╡ 4c5d29e0-5887-11eb-13a8-3fc1fc4ff05e
@show "Error norm gain - HW/FW: $(round(abs(fastr₇₈), digits=5))"

# ╔═╡ dac43802-586a-11eb-3b90-8fbf23bfb42b
# Visualizing solution errors
begin
	gⱼ₃ 	= reshape(eⱼ₇ - eⱼ₈, (n₁, n₁))' |> Array
	htmpⱼ₃ 	= heatmap(1:n₁, 1:n₁, gⱼ₃, fmt=:png, ratio=1,
						title="Approx. Difference - Halfweighting/Fullweighting")
	sfcⱼ₃ 	= surface(gⱼ₃, fmt=:png, ratio=1, legend=false,
						title="Approx. Difference Distribution")
	plot(htmpⱼ₃, sfcⱼ₃, layout=(1,2), size=(1000,500))
end

# ╔═╡ 5ad1d1f4-5881-11eb-2c3c-177e2beef14f
@show "||eᵢ|| $(fastr₆₇ < 0 ? '<' : '>') ||eₕ||"

# ╔═╡ 3790a728-5887-11eb-0d61-651e924d1035
@show "Error norm gain - I/HW: $(round(abs(fastr₆₇), digits=5))"

# ╔═╡ 88d86508-5887-11eb-00b8-79cb3b213e22
# Visualizing solution errors
begin
	gⱼ₂ 	= reshape(eⱼ₆ - eⱼ₇, (n₁, n₁))' |> Array
	htmpⱼ₂ 	= heatmap(1:n₁, 1:n₁, gⱼ₂, fmt=:png, ratio=1,
						title="Approx. Difference - Injection/Halfweighting")
	sfcⱼ₂ 	= surface(gⱼ₂, fmt=:png, ratio=1, legend=false,
						title="Approx. Difference Distribution")
	plot(htmpⱼ₂, sfcⱼ₂, layout=(1,2), size=(1000,500))
end

# ╔═╡ 5ae6f7da-5881-11eb-00eb-732d1dfcf26f
@show "||eᵢ|| $(fastr₆₈ < 0 ? '<' : '>') ||eᵩ||"

# ╔═╡ 494121ac-5887-11eb-1264-7155e70d3f0f
@show "Error norm gain I/FW: $(round(abs(fastr₆₈), digits=5))"

# ╔═╡ 92df302a-5889-11eb-2eeb-e7122d1d6782
md"""
What we learn from these results is that as long as we're not changing the local values of the solution matrices, there's no problem coarsening the grid some more. Which is in a bit the opposite result of the comparison between the *linearization* and the direct *enlargement* prolongations operators. This may be explained by the fact that when using the multigrid solver, we're trying to port our problem into coarser grids to solve it on that level, so if we're changing the values **even if it's a local-aware change** *(as it's the case for halfweighting and even more in fullweighting)*, we may be **changing the target distribution** and by such furthering our path to the optimal minima.

**Anisotropic Problem Resolution**

We already constructed the Anisotropic problem PDE $(P_2)$ in the PDE's & Operators Construction part of the document, so we're going to directly try to solve it using our 4-level 2-step W-cycle multigrid equipped by the injection & linearization interpolation operators.

**1-** $ϵ = 0.3$
"""

# ╔═╡ 9397065a-5889-11eb-0de5-5b56807adc26
begin
	ϵ₁ 	= 0.3
	A₂₁ = A₂(ϵ₁)
	uⱼ₉ = multigrid(A₂₁, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, injection, linearize)
	eⱼ₉ = bₙ₁ - A₂₁(n₁)*uⱼ₉
end ;

# ╔═╡ 93bfc128-5889-11eb-3009-5b8ca0ec96d8
# Visualizing solution errors
begin
	gⱼ₄ 	= reshape(eⱼ₉, (n₁, n₁))' |> Array
	htmpⱼ₄ 	= heatmap(1:n₁, 1:n₁, gⱼ₄, fmt=:png, ratio=1,
						title="Approx. Error - ϵ = 0.3")
	sfcⱼ₄ 	= surface(gⱼ₄, fmt=:png, ratio=1, legend=false,
						title="Approx. Error Distribution")
	plot(htmpⱼ₄, sfcⱼ₄, layout=(1,2), size=(1000,500))
end

# ╔═╡ 692663c8-5890-11eb-2667-53cabbb41ef2
md"""**2-** $ϵ = 3$"""

# ╔═╡ 7d33ca6c-5891-11eb-0f78-d12c4db73a43
begin
	ϵ₂ 		= 3.
	A₂₂ 	= A₂(ϵ₂)
	uⱼ₁₀ 	= multigrid(A₂₂, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, injection, linearize)
	eⱼ₁₀ 	= bₙ₁ - A₂₂(n₁)*uⱼ₁₀
end ;

# ╔═╡ 7d365d0e-5891-11eb-048a-d3cfdbce0aba
# Visualizing solution errors
begin
	gⱼ₅ 	= reshape(eⱼ₁₀, (n₁, n₁))' |> Array
	htmpⱼ₅ 	= heatmap(1:n₁, 1:n₁, gⱼ₅, fmt=:png, ratio=1,
						title="Approx. Error - ϵ = 3")
	sfcⱼ₅ 	= surface(gⱼ₅, fmt=:png, ratio=1, legend=false,
						title="Approx. Error Distribution")
	plot(htmpⱼ₅, sfcⱼ₅, layout=(1,2), size=(1000,500))
end

# ╔═╡ 22af0768-5892-11eb-3954-41099e721288
md"""**3-** $ϵ = 30$"""

# ╔═╡ 22b4d472-5892-11eb-091f-553428e2dc6a
begin
	ϵ₃ 		= 30.
	A₂₃ 	= A₂(ϵ₃)
	uⱼ₁₁ 	= multigrid(A₂₃, bₙ₁, uₙ₁, 4, ω, 1e-30, 2, injection, linearize)
	eⱼ₁₁ 	= bₙ₁ - A₂₃(n₁)*uⱼ₁₁
end ;

# ╔═╡ 22bf4c40-5892-11eb-3e51-075e4309d973
# Visualizing solution errors
begin
	gⱼ₆ 	= reshape(eⱼ₁₁, (n₁, n₁))' |> Array
	htmpⱼ₆ 	= heatmap(1:n₁, 1:n₁, gⱼ₆, fmt=:png, ratio=1,
						title="Approx. Error - ϵ = 30")
	sfcⱼ₆ 	= surface(gⱼ₆, fmt=:png, ratio=1, legend=false,
						title="Approx. Error Distribution")
	plot(htmpⱼ₆, sfcⱼ₆, layout=(1,2), size=(1000,500))
end

# ╔═╡ 93d3d1dc-5889-11eb-3f54-1319a2e33867
md"""
Trying to analyze the figures we get from the 3 simulations we ran, we can identify one main problem which is the **non-convergence on the pure directions** being X and Y axis of the vector space. Also, when we look into the difference between the first and the second graphs, we may conclude that the error is also *alternating from one direction into the other* while moving through intervals of values of $\epsilon$.

This is actually caused by the nature of the **deformation** that the PDE applies to the vector space. As the equation $P_2$ states, differentiating the solution on the X-axis by one unit will directly force differentiating it by $\epsilon$ units onto the Y-axis.

$-\frac{\partial^2 u(x,y)}{∂x²} - \epsilon \frac{\partial^2 u(x,y)}{∂y²} = f(x,y)\ \ \ \ (P_2)$

While this may sound not-harmful at all for our solvers, what we can guess is causing this issue, is actually our coarsening strategies. They're **based on equally distributed spaces** and that's why they're affecting **each one of the neighbors equal coefficients** when using neighbors-based coarsening. And this is the exact reason why the injection is yielding the best restriction therefore the best approximation among all the other operators on this problem.

One new way we can think of solving this problem, is to try to create a *halfweighting-like version of a restriction operator*, but **taking into account the $\epsilon$ parameter for the Y direction**.
"""

# ╔═╡ 9453db7c-5889-11eb-0a4f-63e7ef7bdb6d
begin
	"""
		ϵ_halfweight(ϵ, grid)

	A function that returns the ϵ-halfweighting of the grid taken as input.
	"""
	function ϵ_halfweight(ϵ, grid)
		g = Float64.(grid)
		for i=2:2:size(grid,1)-1, j=2:2:size(grid,2)-1
			g[i,j] = ϵ * g[i,j] / 4 - (
				g[i-1,j] + g[i+1,j]
				+ (1 - ϵ/2) * g[i,j-1] + (1 - ϵ/2) * g[i,j+1]) / 8
		end
		return injection(g)
	end
	
	"""
		ϵ_halfweight(ϵ)

	A function that constructs the ϵ-halfweighting operator.
	"""
	ϵ_halfweight(ϵ) = grid -> ϵ_halfweight(ϵ, grid)
end ;

# ╔═╡ 946796a8-5889-11eb-1ecf-05cc703365a8
begin
	uⱼ₁₂ = multigrid(A₂₂, bₙ₁, uₙ₁, 4, ω, 1e-30, 2,
					 ϵ_halfweight(ϵ₂), enlarge)
	eⱼ₁₂ = bₙ₁ - A₂₂(n₁)*uⱼ₁₂
end ;

# ╔═╡ 96746de0-5889-11eb-155b-4961c27f065f
# Visualizing solution errors
begin
	gⱼ₇ 	= reshape(eⱼ₁₂, (n₁, n₁))' |> Array
	htmpⱼ₇ 	= heatmap(1:n₁, 1:n₁, gⱼ₇, fmt=:png, ratio=1,
						title="ϵ-halfweighting - ϵ = 3")
	sfcⱼ₇ 	= surface(gⱼ₇, fmt=:png, ratio=1, legend=false,
						title="Approx. Error Distribution")
	plot(htmpⱼ₇, sfcⱼ₇, layout=(1,2), size=(1000,500))
end

# ╔═╡ 968522d4-5889-11eb-2a74-39a374ed38ca
norm(eⱼ₁₂, 2) - norm(eⱼ₁₀, 2)

# ╔═╡ b4194c8a-58a7-11eb-18af-ef8859041bd8
md"""
**And voilà!**

We finally obtained an operator that worked better and produced a cleaner approximation with less error norm than the best operator among the restrictors tried on the anisotropic problem.

The idea behind it is that as the Y-axis has got **stretched out by $\epsilon$**, then at every point of our grid, we'll have an $\epsilon$-contribution of one element, and $1-\epsilon/N$, $N$ being the number of neighbors around it that are taken into account.

By such, we could decrease our norm by **55-units** on the euclidian norm of the vector space."""

# ╔═╡ e20eecfe-58ad-11eb-387c-ddbbbdeeedd1
html"""
<p style="text-align:right">
<small>Rami KADER . HPC-AI20</small>
-
<small>Mines ParisTech</small>
<br/>
<small><small>Sat. Jan 16, 2021</small></small>
</p>
"""

# ╔═╡ Cell order:
# ╟─2bf55ba0-554e-11eb-1ba0-3723b6f49d8d
# ╟─f893039e-58be-11eb-0c48-f9f499f6856d
# ╠═861aac40-552b-11eb-1151-6b74f5ed3de5
# ╟─c33cb002-553d-11eb-31d3-3176668c06df
# ╟─c68281e6-5539-11eb-10c0-fd665be52ae0
# ╟─0e90ca56-5849-11eb-15b6-7f4419690f96
# ╟─3bccede2-5849-11eb-21e0-37c579ac744e
# ╟─55b1fd86-584d-11eb-26f3-a739ab86bc4d
# ╟─34f711ea-554f-11eb-342f-0358c848c965
# ╟─ba523a0e-554f-11eb-162f-8b6e6434d21c
# ╟─4f335f12-573b-11eb-042c-a97c20dd929d
# ╟─8a710356-573b-11eb-11b0-1b418859bb13
# ╠═be97faee-5532-11eb-1466-15d0f84888cf
# ╟─011d1544-58de-11eb-31bc-fb0262856677
# ╠═3f38303e-58de-11eb-3bb5-ef7da4dfcce3
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
# ╟─6ce6509e-5783-11eb-1066-515dfad96fec
# ╟─b69d80da-5856-11eb-1fae-5fdf08297723
# ╟─421e9880-585f-11eb-1a4a-db98dbf494d6
# ╟─3a7b6bbe-5853-11eb-12d0-2d3907b68685
# ╟─441ed93e-5860-11eb-07f0-f70c9720c8b5
# ╟─0298049a-57b4-11eb-0cb2-e3524c90e250
# ╟─870b1224-5860-11eb-3d77-1d6a0bc7f8a6
# ╟─71deb1c0-5843-11eb-315c-b9cee350e560
# ╟─32f264d2-584a-11eb-01ba-3b4ea67003df
# ╟─d27ae6f8-586d-11eb-1ed3-1da566271a73
# ╟─d25069b4-5874-11eb-3876-dd2e4b0cce2a
# ╟─b1ee0492-5874-11eb-35ef-27791afce7a1
# ╠═e70c7750-5863-11eb-38ea-157cc0b19d9b
# ╠═35ce219c-5869-11eb-39f9-577d4de271b1
# ╠═5757f624-5869-11eb-3006-6fd9f81ba58c
# ╟─624f194a-5873-11eb-1ed4-112551856dfe
# ╟─ebf9da62-586a-11eb-0386-7576e50ef520
# ╟─04354736-5877-11eb-0a11-dd87cbdf782e
# ╟─4a51a37e-586b-11eb-2304-a73fbb512158
# ╟─69ca47ba-586b-11eb-18a6-8128dd4670ce
# ╟─3944a45c-5873-11eb-1af2-65ad1dbc48df
# ╠═c4d86ad0-586e-11eb-3296-8f61deff1e47
# ╟─1037e1a6-5872-11eb-0820-bd41a05d1af1
# ╟─f0e013b0-586e-11eb-2457-f917877375cb
# ╟─18c95184-5871-11eb-0c6e-d3e710f51be2
# ╟─362bda12-5871-11eb-3c16-8d53b233a01d
# ╟─41c925c2-5877-11eb-2e1d-af5b5f0cd74a
# ╟─3a56d20a-5875-11eb-03c0-a5b112d19efe
# ╠═3a5cb224-5875-11eb-2c1b-97436009b686
# ╟─3a6569f0-5875-11eb-344d-5bf556e0aa8d
# ╟─3a668812-5875-11eb-326f-87f50db349c8
# ╟─3a784ba6-5875-11eb-1979-7d9a299ea243
# ╟─3a805846-5875-11eb-0f01-071787fe43fc
# ╟─446a6f40-586b-11eb-282a-f94ee14269a2
# ╟─1832d306-5878-11eb-311c-11fabb6bbb2c
# ╠═18391a04-5878-11eb-3287-57163da7c4f8
# ╟─183e589a-5878-11eb-3584-b37c8b96fb45
# ╟─18484de4-5878-11eb-22f5-f37516a22b29
# ╟─184a1dc4-5878-11eb-3513-9fe97c35da4c
# ╟─1854a65c-5878-11eb-2473-b7acbd68e216
# ╟─1855c320-5878-11eb-0468-b76ec811f012
# ╟─b03feb94-587c-11eb-04b3-33363338be39
# ╠═b043fe28-587c-11eb-31a6-d9199b0a3231
# ╟─b045b128-587c-11eb-1064-8166fe875bd9
# ╟─b053d5a2-587c-11eb-0fcf-053648eb60d9
# ╟─b0551278-587c-11eb-174f-cf692bcd0675
# ╟─b060aa6e-587c-11eb-3ac9-5935786e20c1
# ╟─b061db8a-587c-11eb-312a-fd93e66e3039
# ╠═5cfbb91a-5881-11eb-39ed-817b986b9832
# ╠═5d1657a0-5881-11eb-2784-0d213e722e7f
# ╠═5d2bbe10-5881-11eb-3b4c-3984a432b8f1
# ╟─1a70c16a-5886-11eb-188e-317a78df7b25
# ╠═5d41df08-5881-11eb-3abc-6f4480396028
# ╟─5afc977c-5881-11eb-3629-ef6b8f815404
# ╟─4c5d29e0-5887-11eb-13a8-3fc1fc4ff05e
# ╟─dac43802-586a-11eb-3b90-8fbf23bfb42b
# ╟─5ad1d1f4-5881-11eb-2c3c-177e2beef14f
# ╟─3790a728-5887-11eb-0d61-651e924d1035
# ╟─88d86508-5887-11eb-00b8-79cb3b213e22
# ╟─5ae6f7da-5881-11eb-00eb-732d1dfcf26f
# ╟─494121ac-5887-11eb-1264-7155e70d3f0f
# ╟─92df302a-5889-11eb-2eeb-e7122d1d6782
# ╟─9397065a-5889-11eb-0de5-5b56807adc26
# ╟─93bfc128-5889-11eb-3009-5b8ca0ec96d8
# ╟─692663c8-5890-11eb-2667-53cabbb41ef2
# ╟─7d33ca6c-5891-11eb-0f78-d12c4db73a43
# ╟─7d365d0e-5891-11eb-048a-d3cfdbce0aba
# ╟─22af0768-5892-11eb-3954-41099e721288
# ╟─22b4d472-5892-11eb-091f-553428e2dc6a
# ╟─22bf4c40-5892-11eb-3e51-075e4309d973
# ╟─93d3d1dc-5889-11eb-3f54-1319a2e33867
# ╠═9453db7c-5889-11eb-0a4f-63e7ef7bdb6d
# ╟─946796a8-5889-11eb-1ecf-05cc703365a8
# ╟─96746de0-5889-11eb-155b-4961c27f065f
# ╠═968522d4-5889-11eb-2a74-39a374ed38ca
# ╟─b4194c8a-58a7-11eb-18af-ef8859041bd8
# ╟─e20eecfe-58ad-11eb-387c-ddbbbdeeedd1
