import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist, pdist

from src.eval import tst_reference_response as frozen

EIGEN_RELATIVE_CUTOFF = 1e-10


def pre_pca(standards, blocks):
    return np.concatenate([(np.asarray(v)-s["mean"])/s["scale"]*s["active"]/np.sqrt(s["dimension"])
                           for s, v in zip(standards, blocks, strict=True)], axis=1)


def bandwidth(values):
    unique = np.unique(values, axis=0)
    distances = pdist(unique, "sqeuclidean")
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else None


def rbf(left, right, median):
    if median is None:
        return np.ones((len(left), len(right)))
    return np.exp(-cdist(left, right, "sqeuclidean")/(2*median))


def kernel_features(source, reference):
    mx = bandwidth(source)
    mr = bandwidth(reference) if reference is not None else None
    kx = rbf(source, source, mx)
    kr = rbf(reference, reference, mr) if reference is not None else np.ones_like(kx)
    matrix = (1+kx)*(1+kr)/4
    eigenvalues, vectors = np.linalg.eigh(matrix)
    threshold = eigenvalues[-1]*EIGEN_RELATIVE_CUTOFF
    assert eigenvalues[0] >= -threshold
    keep = eigenvalues > threshold
    model = {"source": source, "reference": reference, "source_median": mx, "reference_median": mr,
             "eigenvalues": eigenvalues[keep], "vectors": vectors[:, keep],
             "all_eigenvalues": eigenvalues, "rank": int(keep.sum())}
    phi = vectors[:, keep]*np.sqrt(eigenvalues[keep])
    return model, phi


def predict_features(model, source, reference):
    kx = rbf(source, model["source"], model["source_median"])
    kr = rbf(reference, model["reference"], model["reference_median"]) if model["reference"] is not None else np.ones_like(kx)
    return ((1+kx)*(1+kr)/4) @ model["vectors"] / np.sqrt(model["eigenvalues"])


def predict_operators(model, source, reference):
    return np.einsum("nr,rdc->ndc", predict_features(model, source, reference), model["weights"])


def regularizer(dimension, smoothness=1e-6, residual=1e-4):
    d = frozen.second_derivatives(dimension)
    return (smoothness*(d.T@d)/d.shape[0]+sparse.eye(dimension**3)*residual/dimension**3).tocsr()


def sufficient(source, targets, rows, dimension):
    b = frozen.lattice_design(source, dimension)
    return {"rows": rows, "gram": (b.T@b/len(source)).tocsr(),
            "rhs": np.stack([b.T@(y-source)/len(source) for y in targets])}


def pair_matrix(count):
    assert count in (1, 2)
    return np.array([[.75, -.25], [-.25, .75]]) if count == 2 else np.ones((1, 1))


def normal_system(phi, groups, reg, beta):
    n, rank = phi.shape
    dimension = reg.shape[0]
    rhs_rows = np.zeros((n, dimension, 3))
    diagonal = np.full((rank, dimension), beta)
    for group in groups:
        indices = group["rows"]
        coefficients = pair_matrix(len(indices))
        rhs_rows[indices] = np.einsum("ij,jdc->idc", coefficients, group["rhs"])
        local = phi[indices]
        diagonal += np.einsum("ir,ij,jr->r", local, coefficients, local)[:, None]*group["gram"].diagonal()
        diagonal += (local**2).sum(0)[:, None]*reg.diagonal()/len(indices)
    assert np.all(diagonal > 0)
    rhs = np.einsum("nr,ndc->rdc", phi, rhs_rows)

    def matvec(vector):
        w = vector.reshape(rank, dimension)
        u = phi@w
        action = np.zeros_like(u)
        for group in groups:
            indices = group["rows"]
            mixed = pair_matrix(len(indices))@u[indices]
            action[indices] = (group["gram"]@mixed.T).T+(reg@u[indices].T).T/len(indices)
        return (phi.T@action+beta*w).ravel()

    size = rank*dimension
    operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator((size, size), matvec=lambda v: v/diagonal.ravel(), dtype=np.float64)
    return operator, preconditioner, rhs, diagonal


def solve_channel(operator, preconditioner, rhs, rtol=1e-8, maxiter=1500):
    flat = rhs.ravel()
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    values, info = cg(operator, flat, x0=np.zeros_like(flat), rtol=rtol, atol=0., maxiter=maxiter,
                      M=preconditioner, callback=callback)
    error = np.linalg.norm(operator@values-flat)
    norm = np.linalg.norm(flat)
    relative = float(error/norm) if norm else float(error)
    solved = bool(info == 0 and relative <= rtol and np.isfinite(values).all())
    return values.reshape(rhs.shape), {"info": int(info), "iterations": iterations,
                                      "relative_residual": relative, "solved": solved}


def objective_from_statistics(u, groups, reg, weights=None, beta=0.):
    data_without_constant = 0.
    penalty = 0.
    for group in groups:
        indices = group["rows"]
        local = u[indices]
        c = pair_matrix(len(indices))
        applied = np.stack([group["gram"]@v for v in local])
        data_without_constant += np.einsum("idc,ij,jdc->", local, c, applied)
        data_without_constant -= 2*np.einsum("idc,ij,jdc->", local, c, group["rhs"])
        penalty += sum(np.sum(v*(reg@v)) for v in local)/len(indices)
    weight_penalty = beta*float(np.sum(weights**2)) if weights is not None else 0.
    return {"data_without_constant": float(data_without_constant), "grid": float(penalty),
            "weight": weight_penalty,
            "normalized_without_constant": float((data_without_constant+penalty+weight_penalty)/(len(groups)*3))}
