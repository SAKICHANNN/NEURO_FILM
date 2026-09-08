import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator

from src.eval import tst_reference_kernel as original


def block_preconditioner(phi, groups, reg):
    rows, rank = phi.shape
    dimension = reg.shape[0]
    assert sorted(i for g in groups for i in g["rows"]) == list(range(rows))
    collapsed = rank != rows
    if collapsed:
        assert rank == len(groups)
        representatives = [g["rows"][0] for g in groups]
        basis = phi[representatives]
        for g in groups:
            np.testing.assert_allclose(phi[g["rows"]], np.broadcast_to(phi[g["rows"][0]], (len(g["rows"]), rank)), rtol=1e-10, atol=1e-12)
    else:
        basis = phi
    assert basis.shape == (rank, rank) and np.linalg.matrix_rank(basis) == rank
    inverse = np.linalg.solve(basis, np.eye(rank))
    identity_error = float(np.max(np.abs(inverse@basis-np.eye(rank))))
    assert identity_error < 1e-9
    dense_reg = reg.toarray()
    factors = []
    for g in groups:
        gram = g["gram"].toarray()
        factors.append((cho_factor(gram+dense_reg, lower=True),
                        cho_factor(gram+.5*dense_reg, lower=True) if len(g["rows"]) == 2 and not collapsed else None))

    def apply(vector):
        transformed = inverse.T@vector.reshape(rank, dimension)
        solved = np.empty_like(transformed)
        for gi, g in enumerate(groups):
            indices = g["rows"]
            mean_factor, difference_factor = factors[gi]
            if collapsed:
                solved[gi] = cho_solve(mean_factor, transformed[gi])
            elif len(indices) == 1:
                solved[indices[0]] = cho_solve(mean_factor, transformed[indices[0]])
            else:
                a, b = indices
                mean = 2*cho_solve(mean_factor, (transformed[a]+transformed[b])/np.sqrt(2))
                difference = cho_solve(difference_factor, (transformed[a]-transformed[b])/np.sqrt(2))
                solved[a] = (mean+difference)/np.sqrt(2)
                solved[b] = (mean-difference)/np.sqrt(2)
        return (inverse@solved).ravel()

    size = rank*dimension
    return LinearOperator((size, size), matvec=apply, dtype=np.float64), {
        "kind": "collapsed_source_blocks" if collapsed else "full_row_blocks",
        "rank": rank, "rows": rows, "basis_inverse_error": identity_error,
        "basis_condition": float(np.linalg.cond(basis)),
        "claim": "Inverse of H without betaI used only as preconditioner; original H/rhs retained."}


class RecoveryKernel:
    def __init__(self, freegrid, previous_states, rtol=1e-8, maxiter=1500):
        self.freegrid = freegrid
        self.previous_states = previous_states
        self.rtol = rtol
        self.maxiter = maxiter
        self.systems = {}
        self.audit = []

    def __getattr__(self, name):
        return getattr(original, name)

    def normal_system(self, phi, groups, reg, beta):
        operator, diagonal_preconditioner, rhs, diagonal = original.normal_system(phi, groups, reg, beta)
        if beta == 0:
            np.testing.assert_array_equal(phi, np.eye(len(phi)))
            preconditioner = diagonal_preconditioner
            info = {"kind": "reused_freegrid", "channels": 0, "beta": beta}
        else:
            preconditioner, info = block_preconditioner(phi, groups, reg)
            info.update({"channels": 0, "beta": beta})
        self.systems[id(preconditioner)] = info
        self.audit.append(info)
        return operator, preconditioner, rhs, diagonal

    def solve_channel(self, operator, preconditioner, rhs, rtol=1e-8, maxiter=1500):
        assert rtol == self.rtol and maxiter == self.maxiter
        info = self.systems[id(preconditioner)]
        channel = info["channels"]
        assert channel < 3
        info["channels"] += 1
        if info["kind"] != "reused_freegrid":
            values, state = original.solve_channel(operator, preconditioner, rhs, rtol, maxiter)
            return values, {**state, "initialization": "zeros", "reused": False, "preconditioner": info["kind"]}
        assert self.previous_states[channel]["solved"]
        values = self.freegrid[:, :, channel].copy()
        assert values.shape == rhs.shape
        residual = np.linalg.norm(operator@values.ravel()-rhs.ravel())/max(np.linalg.norm(rhs), 1e-300)
        assert residual <= rtol and np.isfinite(values).all()
        assert abs(residual-self.previous_states[channel]["relative_residual"]) < 1e-12
        return values, {"info": 0, "iterations": 0, "previous_iterations": self.previous_states[channel]["iterations"],
                        "relative_residual": float(residual), "solved": True, "reused": True,
                        "origin": "original_failed_attempt/freegrid_only"}
