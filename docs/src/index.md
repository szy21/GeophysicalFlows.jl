# GeophysicalFlows.jl Documentation

## Overview

`GeophysicalFlows.jl` is a collection of modules which leverage the 
[FourierFlows.jl](https://github.com/FourierFlows/FourierFlows.jl) framework to provide
solvers for problems in Geophysical Fluid Dynamics, on periodic domains using Fourier-based pseudospectral methods.


## Examples

Examples aim to demonstrate the main functionalities of each module. Have a look at our Examples collection!


!!! note "Fourier transforms normalization"
    
    Fourier-based pseudospectral methods rely on Fourier expansions. Throughout the 
    documentation we denote symbols with hat, e.g., ``\hat{u}``, to be the Fourier transform 
    of ``u`` like, e.g.,
    
    ```math
    u(x) = \sum_{k_x} \hat{u}(k_x) \, e^{i k_x x} .
    ```
    
    The convention used in the modules is that the Fourier transform of a variable, e.g., `u` 
    is denoted with `uh` (where the trailing `h` is there to imply "hat"). Note, however, 
    that `uh` is obtained via a FFT of `u` and due to different normalization factors that the 
    FFT algorithm uses, `uh` _is not_ exactly the same as ``\hat{u}`` above. Instead,
    
    ```math
    \hat{u}(k_x) = \frac{𝚞𝚑}{n_x e^{- i k_x x_0}} ,
    ```
    
    where ``n_x`` is the total number of grid points in ``x`` and ``x_0`` is the left-most 
    point of our ``x``-grid.
    
    Read more in the FourierFlows.jl Documentation; see 
    [Grids](https://fourierflows.github.io/FourierFlowsDocumentation/stable/grids/) section.


!!! info "Unicode"
    Oftentimes unicode symbols are used in modules for certain variables or parameters. For 
    example, `ψ` is commonly used to denote the  streamfunction of the flow, or `∂` is used 
    to denote partial differentiation. Unicode symbols can be entered in the Julia REPL by 
    typing, e.g., `\psi` or `\partial` followed by the `tab` key.
    
    Read more about Unicode symbols in the 
    [Julia Documentation](https://docs.julialang.org/en/v1/manual/unicode-input/).


## Developers

The development of GeophysicalFlows.jl started by [Navid C. Constantinou](http://www.navidconstantinou.com) and [Gregory L. Wagner](https://glwagner.github.io) during the 21st AOFD Meeting 2017. During the 
course of time various people have contributed to GeophysicalFlows.jl, including 
[Lia Siegelman](https://scholar.google.com/citations?user=BQJtj6sAAAAJ), [Brodie Pearson](https://brodiepearson.github.io), and [André Palóczy](https://scholar.google.com/citations?user=o4tYEH8AAAAJ) (see the [example in FourierFlows.jl](https://fourierflows.github.io/FourierFlowsDocumentation/stable/literated/OneDShallowWaterGeostrophicAdjustment/)).


## Citing

If you use GeophysicalFlows.jl in research, teaching, or other activities, we would be grateful 
if you could mention GeophysicalFlows.jl and cite our paper in JOSS:

Constantinou et al., (2021). GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs & GPUs. _Journal of Open Source Software_, **6(60)**, 3053, doi:[10.21105/joss.03053](https://doi.org/10.21105/joss.03053).

The bibtex entry for the paper is:

```bibtex
@article{GeophysicalFlowsJOSS,
  doi = {10.21105/joss.03053},
  url = {https://doi.org/10.21105/joss.03053},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {60},
  pages = {3053},
  author = {Navid C. Constantinou and Gregory LeClaire Wagner and Lia Siegelman and Brodie C. Pearson and André Palóczy},
  title = {GeophysicalFlows.jl: Solvers for geophysical fluid dynamics problems in periodic domains on CPUs \& GPUs},
  journal = {Journal of Open Source Software}
}
```
