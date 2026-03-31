import WaterLily: ×,norm2
# general Ngon functions
@inbounds @inline hat(v) = v/(√(v'*v)+eps(eltype(v))) # 3.516 ns (0 allocations: 0 bytes)
@inbounds @inline dS(u,v) = 0.5f0(u×v) # 20.667 ns (3 allocations: 96 bytes)
@inbounds @inline d²_fast(el,x) = sum(abs2,x-center(el))
@inbounds @inline d²(el,x) = sum(abs2,x-locate(el,x))
@inbounds @inline normal(el) = hat(dS(el))
@inbounds @inline area(el) = norm2(dS(el))

# triangle function
@inbounds @inline center(tri::GeometryBasics.Ngon{3,T,3}) where T = SVector(sum(tri.points)/3.f0...) #1.696 ns (0 allocations: 0 bytes)
@inbounds @inline dS(tri::GeometryBasics.Ngon{3,T,3}) where T = dS(tri.points[2]-tri.points[1],tri.points[3]-tri.points[1]) #1.784 ns (0 allocations: 0 bytes)

# quad function
# area-weighted center
@inbounds @inline center(quad::GeometryBasics.Ngon{3,T,4}) where T =  ((u,v)=norm2.(subdS(quad));
                                                                       c₁=SVector(sum(quad.points[[1,2,4]])/3.f0...);
                                                                       c₂=SVector(sum(quad.points[[2,3,4]])/3.f0...);
                                                                       (c₁*u+c₂*v)./(u+v.+eps(T)))
# oriented area vector
@inbounds @inline dS(quad::GeometryBasics.Ngon{3,T,4}) where T = sum(subdS(quad))
# winding direction checked and correct
@inbounds @inline subdS(quad::GeometryBasics.Ngon{3,T,4}) where T = (dS(quad.points[2]-quad.points[1],quad.points[4]-quad.points[1]),
                                                                     dS(quad.points[4]-quad.points[3],quad.points[2]-quad.points[3]))

# linear shape function interpolation of the nodal velocity values at point `p`
get_velocity(::GeometryBasics.Ngon{3,T,4},args...) where T = zero(SVector{3,T}) # not implemented
function get_velocity(tri::GeometryBasics.Ngon{3,T,3},vel,p::SVector{3,T}) where T
    dA = SVector{3,T}([sub_area(tri,p,Val{i}()) for i in 1:3])
    return SVector(sum(vel.points.*dA)/sum(dA))
end
@inline sub_area(t,p,::Val{1}) = 0.5f0*√sum(abs2,×(SVector(t[2]-p),SVector(t[3]-p)))
@inline sub_area(t,p,::Val{2}) = 0.5f0*√sum(abs2,×(SVector(p-t[1]),SVector(t[3]-t[1])))
@inline sub_area(t,p,::Val{3}) = 0.5f0*√sum(abs2,×(SVector(t[2]-t[1]),SVector(p-t[1])))

# use divergence theorem to calculate volume of surface mesh
# F⃗⋅k⃗ = -⨕pn⃗⋅k⃗ dS = ∮(C-ρgz)n⃗⋅k⃗ dS = ∫∇⋅(C-ρgzk⃗)dV = ρg∫∂/∂z(ρgzk⃗)dV = ρg∫dV = ρgV #
volume(a::GeometryBasics.Mesh) = mapreduce(T->∮(x->x,T).*dS(T),+,a)
volume(body::MeshBody) = volume(body.mesh)

# integrate a function over the surface mesh
integrate(func::Function, body::MeshBody) = mapreduce(T->∮(func,T).*dS(T),+,body.mesh)

# integrate a function, second order
function ∮(func::Function, dT::GeometryBasics.Ngon{3})
    Int =  func(0.5f0*dT.points[1] + 0.5f0*dT.points[2])
    Int += func(0.5f0*dT.points[2] + 0.5f0*dT.points[3])
    Int += func(0.5f0*dT.points[1] + 0.5f0*dT.points[3])
    return Int/3.f0
end
import WaterLily: interp
#@TODO fix this upstream
# check if the point `x` is inside the array `A`
# inA(x::SVector,A::AbstractArray) = (all(0 .≤ x) && all(x .≤ size(A).-2))
# function WaterLily.interp(x::SVector{D,T}, arr::AbstractArray{T,D}) where {D,T}
#     inA(x, arr) ? WaterLily.interp(x, arr) : zero(T)
# end

function get_p(tri::GeometryBasics.Ngon{3,T,D},p::AbstractArray{T,3},δ,::Val{true}) where {T,D}
    c=center(tri); ds=dS(tri); n=hat(ds)
    ds.*interp(c + δ*n, p)
end
function get_p(tri::GeometryBasics.Ngon{3,T,D},p::AbstractArray{T,3},δ,::Val{false}) where {T,D}
    c=center(tri); ds=dS(tri); n=hat(ds)
    ds.*(interp(c + δ*n, p) - interp(c - δ*n, p))
end

function get_v(tri::GeometryBasics.Ngon{3,T,D},vel,u::AbstractArray{T,4},δ,::Val{true})  where {T,D}
    c=center(tri); ds=dS(tri); n=hat(ds)
    u = get_velocity(tri,u,c)
    v₁ = interp(c + δ*n, u)
    v₂ = interp(c + 2δ*n, u)
    return ds.*(u + v₂ - v₁)/2δ
end
function get_v(tri::GeometryBasics.Ngon{3,T,D},vel,p::AbstractArray{T,4},δ,::Val{false})  where {T,D}
    c=center(tri); ds=dS(tri); n=hat(ds)
    u = get_velocity(tri,u,c)
    for j in [-1,1]
        v₁ = interp(c + j*δ*n, u)
        v₂ = interp(c + 2j*δ*n, u)
    end

    τ = zeros(SVector{N-1,T})
    vᵢ = vᵢ .- sum(vᵢ.*nᵢ)*nᵢ
    for j ∈ [-1,1]
        uᵢ = interp(xᵢ+j*δ*nᵢ,u)
        uᵢ = uᵢ .- sum(uᵢ.*nᵢ)*nᵢ
        τ = τ + (uᵢ.-vᵢ)./δ
    end
    return τ
    ds.*(interp(c + δ*n, u) - interp(c - δ*n, u))
end

@inbounds @inline normal2D(tri::GeometryBasics.Ngon{3}) =  SVector{2}(normal(tri)[1:2])
@inbounds @inline center2D(tri::GeometryBasics.Ngon{3}) = SVector{2}(center(tri)[1:2])

function get_p(tri::GeometryBasics.Ngon{3,T,D},p::AbstractArray{T,2},δ,::Val{true}) where {T,D}
    c=center2D(tri); n=normal2D(tri); ar=area(tri);
    p = ar.*n*interp(c + δ*n, p)
    return SA[p[1],p[2],zero(T)]
end

"""
    forces(a::AbstractBody, flow::Flow; δ=2)

Calculates the forces on the body `a` in the flow `flow` using a distance `δ` to the surface.
Only if the body is a `MeshBody` the forces are calculated.
"""
forces(a::GeometryBasics.Mesh, flow::Flow, δ=1, boundary=Val{true}()) = map(T->get_p(T, flow.p, δ, boundary), a)
forces(body::MeshBody, b::Flow; δ=1) = forces(body.mesh, b, δ, Val{body.boundary}())
forces(::AutoBody, ::Flow; kwargs...) = nothing
function forces(a::WaterLily.SetBody, b::Flow; δ=1)
    fa = forces(a.a, b; δ); isnothing(fa) ? forces(a.b, b; δ) : fa
end
