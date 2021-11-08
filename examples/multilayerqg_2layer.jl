# # Phillips model of Baroclinic Instability
#
#md # This example can be viewed as a Jupyter notebook via [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/literated/multilayerqg_2layer.ipynb).
#
# A simulation of the growth of barolinic instability in the Phillips 2-layer model
# when we impose a vertical mean flow shear as a difference ``\Delta U`` in the
# imposed, domain-averaged, zonal flow at each layer.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add GeophysicalFlows, Plots, Printf"
# ```

# ## Let's begin
# Let's load `GeophysicalFlows.jl` and some other needed packages.
#
using NPZ

using GeophysicalFlows, Plots, Printf

using Random: seed!


# ## Choosing a device: CPU or GPU

dev = CPU()     # Device (CPU/GPU)
nothing # hide


# ## Numerical parameters and time-stepping parameters

n = 128                  # 2D resolution = n²
stepper = "FilteredRK4"  # timestepper
     dt = 5e-3           # timestep
 nsteps = 10000          # total number of time-steps
 nsubs  = 25             # number of time-steps for plotting (nsteps must be multiple of nsubs)
nothing # hide


# ## Physical parameters
L = 2π                   # domain size
μ = 5e-2                 # bottom drag
β = 5                    # the y-gradient of planetary PV
 
nlayers = 2              # number of layers
f₀, g = 1, 1             # Coriolis parameter and gravitational constant
 H = [0.2, 0.8]          # the rest depths of each layer
 ρ = [4.0, 5.0]          # the density of each layer
 
 U = zeros(nlayers) # the imposed mean zonal flow in each layer
 U[1] = 1.0
 U[2] = 0.0
nothing # hide


# ## Problem setup
# We initialize a `Problem` by providing a set of keyword arguments. In this example we don't
# do any dealiasing to our solution by providing `aliased_fraction=0`.
prob = MultiLayerQG.Problem(nlayers, dev;
                            nx=n, Lx=L, f₀=f₀, g=g, H=H, ρ=ρ, U=U, μ=μ, β=β,
                            dt=dt, stepper=stepper, aliased_fraction=0)
nothing # hide

# and define some shortcuts.
sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y
nothing # hide


# ## Setting initial conditions

# Our initial condition is some small-amplitude random noise. We smooth our initial
# condidtion using the `timestepper`'s high-wavenumber `filter`.
#
# `ArrayType()` function returns the array type appropriate for the device, i.e., `Array` for
# `dev = CPU()` and `CuArray` for `dev = GPU()`.

seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * ArrayType(dev)(randn((grid.nx, grid.ny, nlayers)))
q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
q₀  = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(prob, q₀)
nothing # hide


# ## Diagnostics

# Create Diagnostics -- `energies` function is imported at the top.
E = Diagnostic(MultiLayerQG.energies, prob; nsteps=nsteps)
diags = [E] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.
nothing # hide


# ## Output

# We choose folder for outputing `.jld2` files and snapshots (`.png` files).
filepath = "."
plotpath = "./plots_2layer"
plotname = "snapshots"
filename = joinpath(filepath, "2layer.jld2")
nothing # hide

# Do some basic file management
if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end
nothing # hide

# And then create Output
get_sol(prob) = sol # extracts the Fourier-transformed solution
function get_u(prob)
  sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid
  
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l * vars.ψh
  invtransform!(vars.u, vars.uh, params)
  
  return vars.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))
nothing # hide

startwalltime = time()

u_data = zeros(round(Int, nsteps / nsubs)+1, n, n, nlayers)
v_data = zeros(round(Int, nsteps / nsubs)+1, n, n, nlayers)
q_data = zeros(round(Int, nsteps / nsubs)+1, n, n, nlayers)
ψ_data = zeros(round(Int, nsteps / nsubs)+1, n, n, nlayers)

for j = 0:round(Int, nsteps / nsubs)

  u_data[j+1, :, :, :] = prob.vars.u
  v_data[j+1, :, :, :] = prob.vars.v
  q_data[j+1, :, :, :] = prob.vars.q
  ψ_data[j+1, :, :, :] = prob.vars.ψ

  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    
    log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE₁: %.3e, KE₂: %.3e, PE: %.3e, walltime: %.2f min", clock.step, clock.t, cfl, E.data[E.i][1][1], E.data[E.i][1][2], E.data[E.i][2][1], (time()-startwalltime)/60)

    println(log)
  end
  
  stepforward!(prob, diags, nsubs)
  MultiLayerQG.updatevars!(prob) 
end

npzwrite("u_data.npy", u_data)
npzwrite("v_data.npy", v_data)
npzwrite("q_data.npy", q_data)
npzwrite("psi_data.npy", ψ_data)

# ## Save
# Finally, we can save, e.g., the last snapshot via
# ```julia
# savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
# savefig(savename)
# ```
