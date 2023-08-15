# %%
%load_ext autoreload
%autoreload 2

# %%
import os
os.environ["GEOMSTATS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %% 
import polyscope as ps
ps.init()

import numpy as np

# fn to turn square array of coords into a vertex/face set for polyscope
def mesh_to_polyscope(mesh, wrap_x=False, wrap_y=False, reverse_x=False, reverse_y=False):
    n, m, _ = np.array(mesh).shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):

            c1 = [i, j]
            c2 = [(i + 1) % n, j]
            c3 = [(i + 1) % n, (j + 1) % m]
            c4 = [i, (j + 1) % m]

            # print(i, n)
            if (i == n - 1) and reverse_x:
                c2[1] = (-c2[1] - 2) % m
                c3[1] = (-c3[1] - 2) % m
                # c2[1] = (-c2[1] - int(m / 2) - 2) % m
                # c3[1] = (-c3[1] - int(m / 2) - 2) % m
            if (j == m - 1) and reverse_y:
                c3[0] = (-c3[0] - 2) % n
                c4[0] = (-c4[0] - 2) % n
                # c3[0] = (-c3[0] - int(n / 2) - 2) % n
                # c4[0] = (-c4[0] - int(n / 2) - 2) % n

            faces[i, j, 0] = coords[c1[0], c1[1]]
            faces[i, j, 1] = coords[c2[0], c2[1]]
            faces[i, j, 2] = coords[c3[0], c3[1]]
            faces[i, j, 3] = coords[c4[0], c4[1]]

            # if i == (n - 1)

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 4)

    return mesh_, faces_

def curve_connections(n):
    n = jnp.arange(n-1)
    return jnp.stack([
        n, n+1
    ], axis=-1)
# %%
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
# %%
EPSILON = 1e-6
# %%
# Sampling of random matrices. NOT uniform
def sample_chol_mat(rng, n, d, C=None):
    samples = jax.random.normal(rng, shape=(n, int(d*(d+1)/2)))
    samples = samples.at[:, :d].set(jnp.abs(samples[:, :d]))

    if C is not None:
        trace = jnp.sum(samples**2, axis=-1)
        samples = samples * jnp.sqrt(C/trace)[:, None] * jnp.tanh(trace/C)[:, None]

    return samples

def sample_tangent_mat(rng, n, d):
    return jax.random.normal(rng, shape=(n, int(d*(d+1)/2)))
# %%
## Functions for the trace equations

# for trace(exp_L(tX))
def trace_equation(L, X, t, d):
    ltx = X[d:]
    ltl = L[d:]
    dx = X[:d]
    dl = L[:d] 
    c = jnp.sum(ltl**2)
    b = jnp.sum(ltl*ltx)
    a = jnp.sum(ltx**2)

    # mask where we are one of the diags being zero as it causes issues
    # possibly we need to clip away as we want PD not PSD. That or we need to 
    # fix the tangent space issue.
    # TODO: this issue is fine if we make sure the exp never hits the edge, hence
    # commented out.
    exp_factor = dx/dl
    # exp_factor = jnp.where(jnp.isnan(dx/dl), 0, dx/dl)

    return c + 2*b*t +a*t**2 + jnp.sum(
        dl**2 * jnp.exp(2*t*exp_factor)
    )

# Only the polynomial part of the trace eqn (unused)
def trace_equation_poly(L, X, t, d):
    ltx = X[d:]
    ltl = L[d:]
    c = jnp.sum(ltl**2)
    b = jnp.sum(ltl*ltx)
    a = jnp.sum(ltx**2)

    return c + b*t +a*t**2

# Only the exponential part of the trace eqn (unused)
def trace_equation_exp(L, X, t, d):
    dx = X[:d]
    dl = L[:d]

    return jnp.nansum(
        dl**2 * jnp.exp(2*t*dx/dl)
    )

# return the solutions to just the polynomial part
# used as a guess for the newton init
def trace_poly_slns(L, X, C, d):
    ltx = X[d:]
    ltl = L[d:]
    c = jnp.sum(ltl**2) - C
    b = jnp.sum(ltl*ltx)
    a = jnp.sum(ltx**2)

    det = jnp.sqrt(b**2 - 4*a*c)
    x1 = (-b + det) / (2*a)
    x2 = (-b - det) / (2*a)
    # x2 = jnp.where(x2<=0, jnp.inf, x2)
    return jnp.stack(
        [x1, x2], axis=-1
    )

# return the solutions for each of the exp parts
# used as a guess for the newton init
def trace_exp_slns(L, X, C, d):
    dx = X[:d]
    dl = L[:d]
    j = jnp.argmax(dx/dl)
    x = jnp.log(C/(dl**2)) / (2*dx/dl)

    return x

# return guessed slns
def trace_slns(L, X, C, d):
    return jnp.concatenate(
        [trace_exp_slns(L, X, C, d),
        trace_poly_slns(L, X, C, d)], axis=-1
    )

# guess a single solution. Smallest approx one bigger than zero
def trace_sln_guess(L, X, C, d):
    guesses = trace_slns(L, X, C, d)
    guesses = jnp.where((guesses < 0) | jnp.isnan(guesses) , jnp.inf, guesses)
    return jnp.min(guesses)

# trace of a matrix in cholesky form
def trace(L, d):
    return jnp.sum(L**2)
# %%

# Mny vmappings

def double_vmap(fn):
    fn_vmap = jax.vmap(
        fn,
        in_axes=(0,0,None,None),
        out_axes=(0)
    )
    fn_vmap2 = jax.vmap(
        fn_vmap,
        in_axes=(None,None,0,None),
        out_axes=(0)
    )
    return fn_vmap2

trace_vmap = jax.vmap(trace, in_axes=(0, None), out_axes=(0))

trace_exp_slns_vmap = jax.vmap(
    trace_exp_slns,
    in_axes=(0,0,None,None),
    out_axes=(0)
)
trace_poly_slns_vmap = jax.vmap(
    trace_poly_slns,
    in_axes=(0,0,None,None),
    out_axes=(0)
)
trace_slns_vmap = jax.vmap(
    trace_slns,
    in_axes=(0,0,None,None),
    out_axes=(0)
)

trace_equation_vmap2 = jax.vmap(
    jax.vmap(
        trace_equation,
        in_axes=(0,0,0,None),
        out_axes=(0)
    ),
    in_axes=(None,None,1,None),
    out_axes=(1)
)

trace_equation_vmap3 = jax.vmap(
    trace_equation,
    in_axes=(0,0,0,None),
    out_axes=(0)
)

trace_equation_vmap = double_vmap(trace_equation)
trace_equation_poly_vmap = double_vmap(trace_equation_poly)
trace_equation_exp_vmap = double_vmap(trace_equation_exp)

trace_equation_grad = jax.grad(trace_equation, argnums=2)
trace_equation_grad_vmap3 = jax.vmap(
    trace_equation_grad,
    in_axes=(0,0,0,None),
    out_axes=(0)
)
trace_sln_guess_vmap = jax.vmap(
    jax.jit(trace_sln_guess, static_argnums=3), 
    in_axes=(0,0,None,None)
)

# %%
## Manifold functions
# Matrices represented as a vector of each diagonal concatenated, and the zero
# elements removed.
# I.e.
# 1 0 0
# 4 2 0
# 6 5 3

# zero out the lower triangular part
def diag(L, d):
    return L.at[d:].set(0)

# zero out the diagonal
def slt(L, d):
    return L.at[:d].set(0)

# return exp_L(X)
def exp(L, X, d):
    # Clipping prevents hitting the diag = 0 boundary which is not in the manifold
    return slt(L + X, d) + diag(L * jnp.exp(X/L), d).clip(EPSILON)

# parallel transport X from L to K
def parallel_transport(L, X, K, d):
    return slt(X, d) + diag(K*X/L, d)
    
# return the geodesic dist from L to K
def geodesic_dist(L, K, d):
    return jnp.sum((L.at[:d].set(jnp.log(L[:d])) - K.at[:d].set(jnp.log(K[:d])))**2)**0.5
# %%
# newton iterate solutions to f for N steps
def newton(N, x, fun, diff_fun):
    def step(i, x):
        return x - (fun(x) / diff_fun(x))

    x = jax.lax.fori_loop(
        0, n, step, x
    )        

    return x

# approx solve the max trace problem using initial guesses
# and newton iteration for N steps
def trace_condition_t(L, X, C, d, N):
    init_t = trace_sln_guess_vmap(L, X, C, d)
    f = lambda t:trace_equation_vmap3(L, X, t, d) - C
    df = lambda t: trace_equation_grad_vmap3(L, X, t, d)
    return newton(N, init_t, f, df)

# %%
# This cell solves the trace problem for some random L/X and 
# plots the trace equation + the sln found
rng = jax.random.PRNGKey(0)
n = 5
d = 2
xlim=2.5

C=5

(k1, k2, rng) = jax.random.split(rng, 3)

L = sample_chol_mat(k1, n, d)
X = sample_tangent_mat(k2, n, d)
t = jnp.linspace(-xlim,xlim,100)

trace_L = trace_vmap(L, d)
L = L[trace_L < C]
X = X[trace_L < C]

trace_L = trace_vmap(L, d)
teq = trace_equation_vmap(L, X, t, d)

n = L.shape[0]

f = lambda t:trace_equation_vmap3(L, X, t, d) - C
df = lambda t: trace_equation_grad_vmap3(L, X, t, d)
tesm_n = newton(10, trace_sln_guess_vmap(L, X, C, d), f, df)

tesm_n = trace_condition_t(L, X, C, d, 10)

cmap = plt.get_cmap("tab10")
for i in range(n):
    plt.semilogy(t, teq[:, i], color=cmap(i))
    plt.axvline(tesm_n[i], color=cmap(i), linestyle=':')

plt.ylim(1,C*2)
plt.xlim(-xlim, xlim)
plt.axhline(C)
# %%`
# This cell finds the trace solutions for many L/X pairs
# and plots a histogram of the error in the trace. Good for testing Ns
rng = jax.random.PRNGKey(0)
n = 50_000
d = 2
xlim=10

C=5

(k1, k2, rng) = jax.random.split(rng, 3)

L = sample_chol_mat(k1, n, d)
X = sample_tangent_mat(k2, n, d)
t = jnp.linspace(-xlim,xlim,100)

trace_L = trace_vmap(L, d)
L = L[trace_L < C]
X = X[trace_L < C]

trace_L = trace_vmap(L, d)
teq = trace_equation_vmap(L, X, t, d)

n = L.shape[0]

f = lambda t:trace_equation_vmap3(L, X, t, d) - C
df = lambda t: trace_equation_grad_vmap3(L, X, t, d)
tesm_n = newton(10, trace_sln_guess_vmap(L, X, C, d), f, df)

tesm_n = trace_condition_t(L, X, C, d, 10)

approx_n = jnp.abs(f(tesm_n))
print(jnp.min(approx_n), jnp.max(approx_n), jnp.mean(approx_n), jnp.std(approx_n))

plt.hist(jnp.log(approx_n).clip(-65))

# %%
# A bunch of vector manipulation functions

# Levi-cevita SYMBOL (not tensor) in 3d
levicevita3d = jnp.array([
    [
        [0,0,0],
        [0,0,1],
        [0,-1,0]
    ],
        [
        [0,0,-1],
        [0,0,0],
        [1,0,0]
    ],

    [
        [0,1,0],
        [-1,0,0],
        [0,0,0]
    ],
])

# return the riemannian metric tensor at L
def g_mat(L, d):
    return jnp.clip(L, EPSILON).at[d:].set(1)**(-2)

# return the inv riemannian metric tensor at L
def ginv_mat(L, d):
    return jnp.clip(L, EPSILON).at[d:].set(1)**(2)

# compute the inner product of e1, e2 at L
def inner_prod(e1, e2, L, d):
    g = g_mat(L, d)
    return jnp.einsum('i,i,i', e1, e2, g)

# assume 2d, diag metric
# riemannian cross product of e1, e2 at L
# slower, not used
def cross_prod(e1, e2, L):
    g = g_mat(L, 2)
    ginv = ginv_mat(L, 2)
    return jnp.einsum(
        'i,j,k,ijk->k', e1, e2, ginv, levicevita3d
    ) * jnp.sqrt(jnp.prod(g))


# Assume 2d
# 3x basis elements of the 2d subspace of the tangent space parallel to constant trace
# one might be degenerate.
def e1(L):
    tv = jnp.array([L[1], -L[0], 0])
    return tv / jnp.sqrt(inner_prod(tv, tv, L, 2))

def e2(L):
    tv = jnp.array([0, L[2], -L[1]])
    return tv / jnp.sqrt(inner_prod(tv, tv, L, 2))

def e3(L):
    tv = jnp.array([L[2], 0, -L[0]])
    return tv / jnp.sqrt(inner_prod(tv, tv, L, 2))

# compute the vector in the tangent space normal to constant trace
def normal(L):
    nv = jnp.array([L[0]**3, L[1]**3, L[2]])
    return nv / jnp.sqrt(inner_prod(nv, nv, L, 2))

# riemannian reflection of tv in the plane defined by n
def reflect(tv, n, L, d):
    return tv - 2*inner_prod(tv, n, L, d)*n

L = sample_chol_mat(rng, 10, 2)[9]
X = sample_tangent_mat(rng, 10, 2)[0]
# L = jnp.ones((3,))

e1l = e1(L)
e2l = e2(L)
e3l = e3(L)

cp1 = cross_prod(e1l, e2l, L)
cp2 = cross_prod(e2l, e3l, L)
cp3 = cross_prod(e3l, e1l, L)

nv = normal(L)

cp1 = cp1 / jnp.sqrt(inner_prod(cp1, cp1, L, d))
cp2 = cp2 / jnp.sqrt(inner_prod(cp2, cp2, L, d))
cp3 = cp3 / jnp.sqrt(inner_prod(cp3, cp3, L, d))

print(L)
print(e1(L))
print(e2(L))
print(e3(L))
print(inner_prod(e1l, e1l, L, d), inner_prod(e2l, e2l, L, d), inner_prod(e3l, e3l, L, d))
print(inner_prod(e1l, e2l, L, d), inner_prod(e2l, e3l, L, d), inner_prod(e1l, e3l, L, d))

print(cp1, cp2, cp3)
print(nv)
# check the cross prods, normal point in the right direction
print(inner_prod(cp1, cp2, L, d), inner_prod(cp2, cp3, L, d), inner_prod(cp3, cp1, L, d))
print(inner_prod(cp1, nv, L, d), inner_prod(cp2, nv, L, d), inner_prod(cp3, nv, L, d))

# check the cross prods, normal are perp to the cont trace
print("cp1 inner prods: ", inner_prod(e1l, cp1, L, d), inner_prod(e2l, cp1, L, d), inner_prod(e3l, cp1, L, d))
print("cp2 inner prods: ", inner_prod(e1l, cp2, L, d), inner_prod(e2l, cp2, L, d), inner_prod(e3l, cp2, L, d))
print("cp3 inner prods: ", inner_prod(e1l, cp3, L, d), inner_prod(e2l, cp3, L, d), inner_prod(e3l, cp3, L, d))
print("nv inner prods:  ", inner_prod(e1l, nv, L, d), inner_prod(e2l, nv, L, d), inner_prod(e3l, nv, L, d))


eps = 0.01
print(trace(L, d))
# check the tv's do roughly the right thing to the trace when moving in that direction.
print(trace(exp(L, eps*e1l, d), d) - trace(L, d))
print(trace(exp(L, eps*e2l, d), d) - trace(L, d))
print(trace(exp(L, eps*e3l, d), d) - trace(L, d))
print(trace(exp(L, eps*nv, d), d) - trace(L, d))

# %%
# make the quarter sphere constraint
N=100
C = 5
phi = jnp.linspace(0, 0.5*jnp.pi,N)
theta = jnp.linspace(0, jnp.pi, N)
phi, theta = jnp.meshgrid(phi, theta)

x = jnp.sqrt(C) * jnp.sin(theta) * jnp.cos(phi)
y = jnp.sqrt(C) * jnp.sin(theta) * jnp.sin(phi)
z = jnp.sqrt(C) * jnp.cos(theta)

# %%
# Plot some exp maps
L = sample_chol_mat(rng, 10, 2)
X = sample_tangent_mat(rng, 10, 2)
X = X / jax.vmap(inner_prod, in_axes=(0,0,0,None))(X, X, L, d)[:, None]
t = jnp.linspace(0,1,100)

def double_vmap2(fn):
    fn_vmap = jax.vmap(
        fn,
        in_axes=(0,0,None),
        out_axes=(0)
    )
    fn_vmap2 = jax.vmap(
        fn_vmap,
        in_axes=(None,0,None),
        out_axes=(0)
    )
    return fn_vmap2

exp_vmap = double_vmap2(exp)
exp_vmap2 = jax.vmap(
    exp, in_axes=(0,0,None)
)

paths = exp_vmap(L, t[:, None, None]*X[None, :, :], d)

ps.register_surface_mesh("sphere", *mesh_to_polyscope(jnp.stack([x,y,z], axis=-1)), smooth_shade=True)
for i in range(paths.shape[1]):
    line = ps.register_curve_network(f"path_{i}", paths[:,i,:], curve_connections(t.shape[0]))
    line.set_radius(0.1, relative=False)

ps.show()

# %%
# Plot some boundary hits + reflections
C = 5
L = sample_chol_mat(rng, 10, 2)
L = L / jnp.sqrt(jnp.sum(L**2, axis=-1))[:, None]
X = sample_tangent_mat(rng, 10, 2)
# X = jax.vmap(diag, in_axes=(0, None))(X, d) + 1e-5 * jax.vmap(slt, in_axes=(0, None))(X, d)
# X = jax.vmap(slt, in_axes=(0, None))(X, d)

X = X / jnp.sqrt(jax.vmap(inner_prod, in_axes=(0,0,0,None))(X, X, L, d)[:, None])

hitting_t = trace_condition_t(L, X, C, d, 10)

t = jnp.linspace(0,hitting_t,10)

paths = exp_vmap(L, t[:, :, None]*X[None, :, :], d)
end_vecs = jax.vmap(parallel_transport, in_axes=(0,0,0,None))(L, X, paths[-1], d)
normals = jax.vmap(normal)(paths[-1])
reflections = jax.vmap(reflect)(end_vecs, normals, paths[-1]) 

hitting_t2 = trace_condition_t(paths[-1], reflections, C, d, 100)
# hitting_t2 = jnp.clip(hitting_t2, 0, 2)

t2 = jnp.linspace(0,hitting_t2,10)
paths2 = exp_vmap(paths[-1], t2[:, :, None]*reflections[None, :, :], d)

dist = jax.vmap(geodesic_dist, in_axes=(0,0,None))(
    paths2[0],
    paths2[-1],
    d
)

# print(jax.vmap(inner_prod, in_axes=(0,0,0,None))(X, X, paths[0], d))
# print(jax.vmap(inner_prod, in_axes=(0,0,0,None))(end_vecs, end_vecs, paths[-1], d))
# print(jax.vmap(inner_prod, in_axes=(0,0,0,None))(reflections, reflections, paths[-1], d))

# t = jnp.linspace(-xlim,xlim,100)
# teq = trace_equation_vmap(paths[-1], reflections, t, d)
# plt.semilogy(t, teq)
# plt.gca().axhline(C)

ps.remove_all_structures()
constraint = ps.register_surface_mesh("sphere", *mesh_to_polyscope(jnp.stack([x,y,z], axis=-1)), smooth_shade=True)
constraint.set_transparency(0.5)
for i in range(paths.shape[1]):
    line = ps.register_curve_network(f"path_{i}", paths[:,i,:], curve_connections(t.shape[0]))
    line.set_radius(0.05, relative=False)

start_cloud = ps.register_point_cloud("start_points", L)
start_cloud.set_radius(0.06, relative=False)
start_cloud.add_vector_quantity("X vecs", X, enabled=True, length=0.2, radius=0.01, vectortype='ambient')

end_cloud = ps.register_point_cloud("end point", paths[-1])
end_cloud.set_radius(0.06, relative=False)
end_cloud.add_vector_quantity("X vecs", end_vecs, enabled=True, length=0.2, radius=0.01, vectortype='ambient')
end_cloud.add_vector_quantity("normals", normals, enabled=True, length=0.2, radius=0.01, vectortype='ambient')
end_cloud.add_vector_quantity("reflections", reflections, enabled=True, length=0.2, radius=0.01, vectortype='ambient')

for i in range(paths2.shape[1]):
    line2 = ps.register_curve_network(f"path2_{i}", paths2[:,i,:], curve_connections(t2.shape[0]))
    line2.set_radius(0.05, relative=False)
ps.show()

# %%
# functions to do the bounded step / reflected exp / brownian sampling
from functools import partial

from diffrax.misc import bounded_while_loop


# @partial(jax.jit, static_argnums=(3,4))
def trace_hitting_distance(L, X, C, d, N):
    init_t = trace_sln_guess_vmap(L, X, C, d)
    f = lambda t:trace_equation_vmap3(L, X, t, d) - C
    df = lambda t: trace_equation_grad_vmap3(L, X, t, d)
    return newton(N, init_t, f, df)

# @partial(jax.jit, static_argnums=(3,4,5))
def bounded_step(base_point, step_dir, step_size, max_trace, newton_steps, d):
    """
        base_point: array of points in S^+ (cholesky) space.
        step_dir: array of unit norm vectors at base_point denoting the direction to go.
        step_size: distance of step to take. assumed > 0.
        max_trace: max trace condition for the boundary.
        newton_steps: number of steps of newtons method to take in solving the distance to the boundary along the geodesic.
        d: dimension of the space (d*d matrices)
    """
    hitting_dist = trace_hitting_distance(base_point, step_dir, C, d, newton_steps)
    step_dist = jnp.clip(step_size, 0, hitting_dist)
    remaining_dist = step_size - step_dist

    next_point = exp_vmap2(base_point, step_dir * step_dist[:, None], d)
    return next_point, remaining_dist

# @partial(jax.jit, static_argnums=(2))
def reflect_tangent(base_point, tangent_vec, d):
    normal_vec = normal(base_point)
    return reflect(tangent_vec, normal_vec, base_point, d)

# @partial(jax.jit, static_argnums=(3,4,5))
def reflected_exp(base_point, tangent_vec, max_trace, d, max_steps, newton_steps):
    def cond_fun(val):
        _, _, remaining_step = val
        return jnp.any(remaining_step > 0)

    def body_fun(val, _):
        base_point, step_dir, step_size = val
        new_base_point, step_size = bounded_step(base_point, step_dir, step_size, max_trace, newton_steps, d)
        new_step_dir = jax.vmap(
            parallel_transport, in_axes=(0,0,0,None)
        )(base_point, step_dir, new_base_point, d)
        new_step_dir = jax.vmap(reflect_tangent, in_axes=(0,0,None))(new_base_point, new_step_dir, d)
        return (new_base_point, new_step_dir, step_size)


    step_size = jnp.sqrt(
        jax.vmap(inner_prod, in_axes=(0,0,0,None))(tangent_vec, tangent_vec, base_point, d)
    )
    step_dir = tangent_vec / step_size[:, None]

    base_point, _, _ = bounded_while_loop(
        cond_fun, 
        body_fun, 
        (base_point, step_dir, step_size), 
        max_steps
    )

    return base_point

def sample_tangent_gaussian(key, L, d):
    g_half = jnp.sqrt((L**(-2)).at[d:].set(1))
    return jax.random.normal(key, L.shape) * g_half / jnp.sqrt(3)

def run_brownain(rng, L, C, d, gamma, steps, newton_steps):
    def loop_fn(i, val):
        rng, L = val
        key, rng = jax.random.split(rng)
        X = jnp.sqrt(gamma) * sample_tangent_gaussian(key, L, d)
        L = reflected_exp(
            L, X, C, d, 1000, newton_steps
        )
        return (rng, L)
    
    return jax.lax.fori_loop(0, steps, loop_fn, (rng, L))[1]


reflected_exp = jax.jit(reflected_exp, static_argnums=(3,4,5))
run_brownain = jax.jit(run_brownain, static_argnums=(3,5,6))
# %%
# plot some reflected exp trajectories
C = 5
L = sample_chol_mat(rng, 10, 2)[1]
L = L / jnp.sqrt(jnp.sum(L**2, axis=-1))
X = sample_tangent_mat(rng, 10, 2)[1]

n_step = 100
t_max = 5
X = X / jnp.sqrt(inner_prod(X, X, L, d))
t = jnp.linspace(0.001, t_max, n_step)
X = t[:, None] * X[None, :]
L = jnp.repeat(L[None, :], n_step, axis=0)

points = reflected_exp(L, X, 5, d, 1000, 10)

ps.remove_all_structures()
ps.register_surface_mesh("sphere", *mesh_to_polyscope(jnp.stack([x,y,z], axis=-1)), smooth_shade=True)

line = ps.register_curve_network(f"path", points, curve_connections(points.shape[0]))
line.set_radius(0.05, relative=False)

ps.show()

# %%