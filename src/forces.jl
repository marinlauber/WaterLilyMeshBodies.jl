"""
    SurfaceForces

Hold the pressure (`pressure`) and viscous (`viscous`) surface forces on the `MeshBody`.
"""
struct SurfaceForces{T,A}
    pressure :: A
    viscous :: A
    function SurfaceForces(body::MeshBody)
        T = typeof(body.scale)
        mem = typeof(body.mesh).name.wrapper
        A = zeros(T,length(body.mesh),3) |> mem
        new{T,typeof(A)}(A,copy(A))
    end
end
export SurfaceForces

function WaterLily.pressure_force(a::SurfaceForces,sim::AbstractSimulation;kwargs...)
    Tp = eltype(a.pressure); To = promote_type(Float64,Tp)
    surface_pressure!(a,sim;kwargs...); sum(To,a.pressure,dims=1)[:] |> Array
end
function surface_pressure!(a::SurfaceForces,sim::AbstractSimulation;δ,boundary=Val{sim.body.boundary}())
    @WaterLily.loop a.pressure[I,:] .= get_p(sim.body.mesh[I],sim.flow.p,δ,boundary) over I in CartesianIndices(1:size(a.pressure,1))
end

function WaterLily.viscous_force(a::SurfaceForces,sim::AbstractSimulation;kwargs...)
    Tp = eltype(a.viscous); To = promote_type(Float64,Tp)
    surface_shear!(a,sim;kwargs...); sum(To,a.viscous,dims=1)[:] |> Array
end
function surface_shear!(a::SurfaceForces,sim::AbstractSimulation;δ,boundary=Val{sim.body.boundary}())
    @WaterLily.loop a.viscous[I,:] .= get_v(sim.body.mesh[I],sim.body.velocity[I],sim.flow.u,δ,boundary) over I in CartesianIndices(1:size(a.viscous,1))
end

import WaterLily: interp
@inline function get_p(tri::SMatrix{3,3,T},p::AbstractArray{T,3},δ,::Val{true}) where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    return ds.*interp(c + δ*n, p) # only outside
end
@inline function get_p(tri::SMatrix{3,3,T},p::AbstractArray{T,3},δ,::Val{false}) where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    return ds.*(interp(c + δ*n, p) - interp(c - δ*n, p)) # both sides
end

@fastmath @inline proj(a,n) = a .- sum(a.*n)*n # tangent component
@inline function get_v(tri::SMatrix{3,3,T},vel,u::AbstractArray{T,4},δ,::Val{true})  where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    vₑ = get_velocity(c,tri,vel)
    v₁ = interp(c + δ*n, u)
    v₂ = interp(c + 2δ*n, u)
    return proj(ds.*(vₑ + v₂ - v₁)/2δ,n) # only outside, projects once
end
@inline function get_v(tri::SMatrix{3,3,T},vel,u::AbstractArray{T,4},δ,::Val{false})  where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    vₑ = get_velocity(c,tri,vel)
    τ = zeros(SVector{3,T})
    for j ∈ [-1,1]
        v₁ = interp(c + j*δ*n, u)
        v₂ = interp(c + j*2δ*n, u)
        τ = τ + ds.*(vₑ + v₂ - v₁)/2δ
    end
    return proj(τ,n) # both sides, projects once
end
