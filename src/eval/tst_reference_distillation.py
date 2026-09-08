import copy

import numpy as np
import torch
from scipy import sparse
from torch import nn

from src.eval import tst_reference_response as frozen


class Encoder(nn.Module):
    def __init__(self, input_dimension=3998, hidden=64, code=16, epsilon=1e-12):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dimension, hidden), nn.GELU(), nn.Linear(hidden, code))
        self.epsilon = epsilon

    def forward(self, values):
        raw = self.layers(values)
        return raw / torch.sqrt(raw.square().sum(-1, keepdim=True) + self.epsilon)


class Decoder(nn.Module):
    def __init__(self, source_dimension=6, code=16, grid=7):
        super().__init__()
        self.grid = grid
        self.linear = nn.Linear((source_dimension+1)*(code+1), 3*grid**3, bias=False)

    def forward(self, source, code):
        s = torch.cat([torch.ones_like(source[:, :1]), source], -1)
        z = torch.cat([torch.ones_like(code[:, :1]), code], -1)
        return self.linear((s[:, :, None]*z[:, None, :]).flatten(1)).reshape(-1, self.grid**3, 3)


def initialize(config):
    torch.manual_seed(config['seed'])
    a = config['architecture']
    teacher = Encoder(2*a['feature'], a['hidden'], a['code'], a['epsilon'])
    decoder = Decoder(a['source'], a['code'], a['grid'])
    student = Encoder(a['feature'], a['hidden'], a['code'], a['epsilon'])
    return teacher, decoder, student, copy.deepcopy(decoder.state_dict()), copy.deepcopy(student.state_dict())


def paired_code(encoder, before, after):
    return encoder(torch.cat([after, after-before], -1))


def after_only(encoder, decoder, source_scores, reference_features):
    assert source_scores.ndim == reference_features.ndim == 2
    assert len(source_scores) == len(reference_features)
    return decoder(source_scores, encoder(reference_features))


def row_weights(group_ids):
    groups = list(dict.fromkeys(group_ids))
    return np.array([1/(len(groups)*group_ids.count(g)) for g in group_ids])


def torch_sparse(matrix):
    matrix = matrix.tocoo()
    return torch.sparse_coo_tensor(np.stack([matrix.row, matrix.col]), matrix.data,
                                   matrix.shape, dtype=torch.float64, check_invariants=True).coalesce()


class PixelQuadratic:
    def __init__(self, sources, targets, group_ids, dimension, paired):
        assert len(sources) == len(targets) == len(group_ids)
        count, nodes = len(sources), dimension**3
        blocks, rhs = {}, np.zeros((count, nodes, 3))
        constant = 0.
        groups = list(dict.fromkeys(group_ids))
        for group in groups:
            ids = [i for i, g in enumerate(group_ids) if g == group]
            assert len(ids) in (1, 2)
            if paired:
                assert all(np.array_equal(sources[ids[0]], sources[i]) for i in ids)
                c = np.array([[.75, -.25], [-.25, .75]]) if len(ids) == 2 else np.ones((1, 1))
                design = frozen.lattice_design(sources[ids[0]], dimension)
                gram = design.T@design/len(sources[ids[0]])
                delta = np.stack([targets[i]-sources[i] for i in ids])
                b = np.stack([design.T@v/len(v) for v in delta])
                rhs[ids] = np.einsum('ij,jdc->idc', c, b)
                constant += np.einsum('ipc,ij,jpc->', delta, c, delta)/delta.shape[1]
                for ai, i in enumerate(ids):
                    for aj, j in enumerate(ids):
                        blocks[i, j] = c[ai, aj]*gram
            else:
                for i in ids:
                    b = frozen.lattice_design(sources[i], dimension)
                    delta = targets[i]-sources[i]
                    blocks[i, i] = b.T@b/(len(delta)*len(ids))
                    rhs[i] = b.T@delta/(len(delta)*len(ids))
                    constant += np.sum(delta**2)/(len(delta)*len(ids))
        zero = sparse.csr_matrix((nodes, nodes))
        matrix = sparse.bmat([[blocks.get((i, j), zero) for j in range(count)] for i in range(count)], format='csr')
        self.matrix = torch_sparse(matrix/(len(groups)*3))
        self.rhs = torch.as_tensor(rhs/(len(groups)*3), dtype=torch.float64)
        self.constant = constant/(len(groups)*3)

    def __call__(self, operators):
        u = operators.to(torch.float64)
        action = torch.sparse.mm(self.matrix, u.reshape(-1, 3)).reshape_as(u)
        return (u*action-2*u*self.rhs).sum()+self.constant


def smoothness(operators, weights, dimension):
    u = operators.to(torch.float64).reshape(-1, dimension, dimension, dimension, 3)
    differences = [torch.diff(u, n=2, dim=axis).flatten(1) for axis in (1, 2, 3)]
    per_row = torch.cat(differences, 1).square().mean(1)*(dimension-1)**4
    return (per_row*weights).sum()


def distillation_loss(zr, zy, teacher_r, teacher_y, weights):
    difference = ((zr-teacher_r.detach()).square().sum(-1)+(zy-teacher_y.detach()).square().sum(-1))/2
    return (difference*weights).sum()


def objective(encoder, decoder, data, query_loss, reference_loss, weights, config, paired=False, teacher_codes=None):
    if paired:
        zr = paired_code(encoder, data['P'], data['R'])
        zy = paired_code(encoder, data['X'], data['Y'])
    else:
        zr, zy = encoder(data['R']), encoder(data['Y'])
    ur, uy = decoder(data['sX'], zr), decoder(data['sX'], zy)
    up = decoder(data['sP'], zr)
    pixel = .5*query_loss(ur)+.5*query_loss(uy)+reference_loss(up)
    smooth = sum(smoothness(u, weights, config['architecture']['grid']) for u in (ur, uy, up))/3
    kd = pixel.new_zeros(())
    if config['kd_weight']:
        assert teacher_codes is not None and not paired
        kd = distillation_loss(zr, zy, *teacher_codes, weights)
    total = pixel+config['smoothness']*smooth+config['kd_weight']*kd
    return total, {'pixel': pixel.detach(), 'smoothness': smooth.detach(), 'kd': kd.detach()}
