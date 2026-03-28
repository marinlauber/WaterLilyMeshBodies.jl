# Signed distance field and measure functions

using StaticArrays
using ForwardDiff
using WaterLily

# signed distance function
WaterLily.sdf(body::Meshbody,x,t;kwargs...) = measure(body,x,t;kwargs...)[1]

# measure
function WaterLily.measure(body::Meshbody,x::SVector{D,T},t;fastd²=Inf) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # before we try the bvh
    !inside(ξ,body.bvh.nodes[1]) && return (T(4),zero(x),zero(x))
    # locate the point on the mesh
    d²,u = closest(ξ,body.bvh,body.mesh;a = body.boundary ? floatmax(T) : T(16))
    u==0 && return (T(4),zero(x),zero(x)) # no triangles within distance "a"
    # compute the normal and distance
    n,p = normal(body.mesh[u]),SVector(locate(ξ,body.mesh[u]))
    # signed Euclidian distance
    d = copysign(√d²,n'*(ξ-p))
    !body.boundary && (d = abs(d)-body.half_thk) # if the mesh is not a boundary, we need to adjust the distance
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    # mesh deformation velocity
    v = get_velocity(p, body.mesh[u], body.velocity[u])
    return (d,dξdx\n,dξdx\dξdt+v)
end
