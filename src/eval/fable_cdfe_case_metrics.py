import numpy as np

from src.eval.fable_photometry_metrics import histogram_errors
from src.eval.fable_reference_photometry import transform


def paired_case_metrics(parameters: dict[str, np.ndarray], ideal: np.ndarray,
                        query_counts: list[np.ndarray], *, slope_limit: float,
                        offset_limit: float) -> dict:
    ideal = np.asarray(ideal, dtype=np.float64)
    if ideal.ndim != 2 or ideal.shape[1:] != (4,) or not len(ideal) or not parameters:
        raise ValueError('nonempty case-by-four ideal parameters and methods required')
    if not query_counts:
        raise ValueError('query histograms required')
    identity = np.repeat((np.arange(256)/255)[:, None], 3, axis=1)
    bounds = dict(slope_limit=slope_limit, offset_limit=offset_limit)
    oracle_tables = np.stack([transform(identity, u, **bounds) for u in ideal])
    change = np.array([[histogram_errors(table, identity, counts)['code_mse']
                        for counts in query_counts] for table in oracle_tables])
    methods = {}
    for name, values in parameters.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != ideal.shape:
            raise ValueError('every method must cover identical ordered cases')
        tables = np.stack([transform(identity, u, **bounds) for u in values])
        error = np.array([[histogram_errors(table, oracle, counts)['code_mse']
                           for counts in query_counts]
                          for table, oracle in zip(tables, oracle_tables, strict=True)])
        output_change = np.array([[histogram_errors(table, identity, counts)['code_mse']
                                  for counts in query_counts] for table in tables])
        methods[name] = {'E': error, 'P': output_change, 'tables': tables}
    return {'D': change, 'oracle_tables': oracle_tables, 'methods': methods}
