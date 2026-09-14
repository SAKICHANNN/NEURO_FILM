from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


DEFAULT_COUNTS = {
    "fit": 256,
    "validation": 32,
    "represented_assessment": 16,
    "held_cameras": 4,
    "assessment_per_held_camera": 4,
    "queries": 4,
}


def solve_role_assignment(
    rows: Sequence[Mapping],
    camera_ranks: Mapping[str, str],
    query_masks: Mapping[str, int],
    counts: Mapping[str, int] | None = None,
    required_query_mask: int = 15,
    block_size: int = 16,
) -> dict | None:
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(
        type(v) is not int or v < 0 for v in counts.values()
    ):
        raise ValueError("Role counts must be complete nonnegative integers")
    if type(required_query_mask) is not int or not 0 <= required_query_mask <= 15:
        raise ValueError("Required query mask must be a four-bit integer")
    if type(block_size) is not int or not 1 <= block_size <= 20:
        raise ValueError("Block size must be in 1..20")
    rows = sorted(rows, key=lambda row: row["candidate_rank"])
    identities = [row["identity"] for row in rows]
    ranks = [row["candidate_rank"] for row in rows]
    if len(set(identities)) != len(rows) or len(set(ranks)) != len(rows):
        raise ValueError("Each admitted component needs a unique identity and rank")
    if any(not isinstance(rank, str) or not rank for rank in ranks):
        raise ValueError("Component ranks must be nonempty strings")
    cameras = {row["camera"] for row in rows}
    if not cameras <= camera_ranks.keys():
        raise ValueError("Every camera needs a frozen rank")
    if any(not isinstance(camera_ranks[c], str) or not camera_ranks[c] for c in cameras):
        raise ValueError("Camera ranks must be nonempty strings")
    if len({camera_ranks[c] for c in cameras}) != len(cameras):
        raise ValueError("Camera ranks must be unique")
    cameras = sorted(cameras, key=camera_ranks.__getitem__)
    if set(query_masks) != set(identities) or any(
        type(mask) is not int or not 0 <= mask <= 15 for mask in query_masks.values()
    ):
        raise ValueError("Every component needs an explicitly bound four-bit query mask")
    if not rows:
        if any(counts.values()) or required_query_mask:
            return None
        return {"held_cameras": [], "fit": [], "validation": [],
                "represented_assessment": [], "held_assessment": {}, "queries": []}

    n, m = len(rows), len(cameras)
    size = 5 * n + 2 * m
    fit, val, rep, held, query = [np.arange(k * n, (k + 1) * n) for k in range(5)]
    camera_held = np.arange(5 * n, 5 * n + m)
    camera_fit = np.arange(5 * n + m, size)
    coefficients, row_indices, column_indices, lower, upper = [], [], [], [], []

    def constrain(entries, lo, hi):
        r = len(lower)
        for col, value in entries:
            column_indices.append(int(col))
            row_indices.append(r)
            coefficients.append(value)
        lower.append(lo)
        upper.append(hi)

    for i in range(n):
        constrain(((k * n + i, 1) for k in range(5)), 0, 1)
    for indices, count in ((fit, counts["fit"]), (val, counts["validation"]),
                           (rep, counts["represented_assessment"]),
                           (query, counts["queries"]), (camera_held, counts["held_cameras"])):
        constrain(((i, 1) for i in indices), count, count)
    camera_members = {}
    for j, camera in enumerate(cameras):
        members = [i for i, row in enumerate(rows) if row["camera"] == camera]
        camera_members[camera] = members
        constrain([(held[i], 1) for i in members] +
                  [(camera_held[j], -counts["assessment_per_held_camera"])], 0, 0)
        constrain([(fit[i], 1) for i in members] + [(camera_fit[j], -len(members))], -np.inf, 0)
        constrain([(fit[i], 1) for i in members] + [(camera_fit[j], -1)], 0, np.inf)
        constrain([(val[i], 1) for i in members] + [(rep[i], 1) for i in members] +
                  [(camera_fit[j], -len(members))], -np.inf, 0)
        constrain([(camera_fit[j], 1), (camera_held[j], 1)], 0, 1)
    for bit in range(4):
        if required_query_mask & (1 << bit):
            constrain(((query[i], 1) for i, identity in enumerate(identities)
                       if query_masks[identity] & (1 << bit)), 1, np.inf)
    matrix = coo_matrix((coefficients, (row_indices, column_indices)),
                        shape=(len(lower), size)).tocsc()
    lower, upper = np.asarray(lower), np.asarray(upper)
    bound_lo, bound_hi = np.zeros(size), np.ones(size)
    constraints = LinearConstraint(matrix, lower, upper)

    def optimize(objective):
        result = milp(objective, integrality=np.ones(size),
                      bounds=Bounds(bound_lo, bound_hi), constraints=constraints,
                      options={"mip_rel_gap": 0.0})
        if result.status == 2:
            return None
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"Assignment optimum not established: {result.message}")
        solution = np.rint(result.x).astype(np.int64)
        values = matrix @ solution
        if (np.max(np.abs(solution - result.x)) > 1e-5 or
                np.any(solution < bound_lo) or np.any(solution > bound_hi) or
                np.any(values < lower) or np.any(values > upper)):
            raise RuntimeError("Solver returned a constraint-invalid integer assignment")
        integer_objective = int(objective @ solution)
        dual_bound = getattr(result, "mip_dual_bound", None)
        if (dual_bound is None or not np.isfinite(dual_bound) or
                dual_bound <= integer_objective - 1 + 1e-5):
            raise RuntimeError("Solver bound does not rule out a better integer objective")
        return solution

    solution = optimize(np.zeros(size))
    if solution is None:
        return None

    def freeze_lex(indices, count):
        nonlocal solution
        indices = list(indices)
        for start in range(0, len(indices), block_size):
            remaining = indices[start:]
            needed = count - int(solution[indices[:start]].sum())
            if needed in (0, len(remaining)):
                bound_lo[remaining] = bound_hi[remaining] = int(needed > 0)
                solution[remaining] = int(needed > 0)
                break
            block = indices[start:start + block_size]
            objective = np.zeros(size)
            # Fixed-cardinality sorted-rank lex order equals reverse membership lex order.
            objective[block] = [-2 ** (len(block) - 1 - k) for k in range(len(block))]
            solution = optimize(objective)
            if solution is None:
                raise RuntimeError("A previously feasible lexicographic prefix became infeasible")
            bound_lo[block] = bound_hi[block] = solution[block]

    freeze_lex(camera_held, counts["held_cameras"])
    chosen_cameras = [camera for j, camera in enumerate(cameras) if solution[camera_held[j]]]
    for indices, name in ((fit, "fit"), (val, "validation"), (rep, "represented_assessment")):
        freeze_lex(indices, counts[name])
    for camera in chosen_cameras:
        freeze_lex([held[i] for i in camera_members[camera]], counts["assessment_per_held_camera"])
    freeze_lex(query, counts["queries"])
    return {
        "held_cameras": chosen_cameras,
        "fit": [identities[i] for i in range(n) if solution[fit[i]]],
        "validation": [identities[i] for i in range(n) if solution[val[i]]],
        "represented_assessment": [identities[i] for i in range(n) if solution[rep[i]]],
        "held_assessment": {camera: [identities[i] for i in camera_members[camera] if solution[held[i]]]
                            for camera in chosen_cameras},
        "queries": [identities[i] for i in range(n) if solution[query[i]]],
    }
