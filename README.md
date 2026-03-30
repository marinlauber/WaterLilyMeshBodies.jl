# WaterLilyMeshBodies

[![CI](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/WaterLily-jl/WaterLilyMeshBodies.jl/actions/workflows/CI.yml)

![dolphin](example/dolphin.png)

[WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) can simulate flow around any body defined by a signed-distance function (SDF), but efficiently computing the SDF from a large surface mesh is non-trivial. The `WaterLilyMeshBodies` package defines a `MeshBody` type and a `measure(body::MeshBody,x::SVector,t::Real)` function that computes the signed distance, surface normal, and surface velocity as needed for a WaterLily simulation. The function runs in $O(\log N)$ time through the use of a Bounding Volume Hierarchy, and works on any backend (single or multi-threaded CPU and GPU).

### Installation

The packages is registered, so you can add it simply using 

```julia
] add WaterLilyMeshBodies
```

### Usage

#### Static mesh body

The simplest way to initialize a `MeshBody` is using an [stl file](https://en.wikipedia.org/wiki/STL_(file_format)) describing the body as a triangle mesh
```julia
using WaterLily, WaterLilyMeshBodies, StaticArrays, CUDA

L = 64
x₀ = SA[L÷4, L÷2, L÷2]
body = MeshBody("path/to/body.stl";
    scale = Float32(L),      # scale mesh to simulation units
    map = (x,t) -> x - x₀,  # centre the body in the domain
    boundary = true,         # closed surface (determines sign of SDF)
    mem = CUDA.CuArray)      # run on GPU
```

Once the body is defined, it can be passed to the WaterLily `Simulation` constructor and used as normal
```julia 
sim = Simulation((2L,L,L), (1,0,0), L; body, ν=1e-3, mem=CUDA.CuArray)
sim_step!(sim, 1.0, remeasure=false) # simulate flow around the static mesh
```

Key keyword arguments to `MeshBody`:

| Argument | Default | Description |
|---|---|---|
| `scale` | `1f0` | Scale factor applied to mesh coordinates |
| `map(x,t)` | identity | Coordinate mapping from simulation space to mesh space |
| `boundary` | `false` | `true` for closed watertight surfaces; `false` for open/shell surfaces |
| `half_thk` | `1.866f0` | Half-thickness (in grid cells) when `boundary=false` |
| `mem` | `Array` | Memory backend: `Array`, `CUDA.CuArray`, etc. |
| `primitive` | `BBox` | Bounding volume type for the BVH: `BBox` (default) or `BSphere` |

#### Deforming mesh body

To translate or deform a mesh, call `update!` each time step with the new triangle coordinates and the time step size:

```julia
update!(body, new_mesh, dt)  # updates positions and derives surface velocity
```

Vertex velocities are computed as $(x_\text{new} - x_\text{old})/\Delta t$ and are used by WaterLily to apply the no-slip boundary condition on moving surfaces.

### Method

Closest-triangle queries are accelerated by a Bounding Volume Hierarchy (BVH) built with [ImplicitBVH.jl](https://github.com/JuliaArrays/ImplicitBVH.jl), reducing each query from $O(N)$ to $O(\log N)$ where $N$ is the number of triangles. The default `boundary=false` results in a open/shell body with finite half-thickness `half_thk`. Setting `boundary=true` attempts to create a closed body signed distance function with $d<0$ internally. The sign is determined by the dot product of the query-to-surface vector and the outward triangle normal, but this can fail when meshes are not "water-tight" or sufficiently irregular, so check the SDF and other `measure`d properties carefully before simulating. The surface velocity at the closest point is interpolated from the triangle's vertex velocities using barycentric shape functions.

