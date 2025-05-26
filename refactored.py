import sys
import numpy as np


def read_matrix(rows, cols, inp):
    matrix = []
    for _ in range(rows):
        matrix.append(list(map(float, inp.pop(0).split())))
    return np.array(matrix)


class Node:
    def __init__(self, op, args):
        self.op = op
        self.args = args
        self.value = None
        self.grad = None


def rlu(x, alpha):
    return np.where(x > 0, x, x * alpha)


def rlu_grad(x, alpha):
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    grad[np.isclose(x, 0)] = 1.0
    return grad


def forward_tnh(node, nodes):
    x = nodes[node.args[0]].value
    node.value = np.tanh(x)
    return node.value.shape

def forward_rlu(node, nodes):
    alpha, xidx = node.args
    x = nodes[xidx].value
    node.value = rlu(x, alpha)
    return node.value.shape

def forward_mul(node, nodes):
    a, b = node.args
    A = nodes[a].value
    B = nodes[b].value
    node.value = np.dot(A, B)
    return node.value.shape

def forward_sum(node, nodes):
    us = node.args
    if len(us) == 1:
        node.value = nodes[us[0]].value.copy()
    else:
        node.value = sum(nodes[u].value for u in us)
    return node.value.shape

def forward_had(node, nodes):
    us = node.args
    v = np.copy(nodes[us[0]].value)
    for u in us[1:]:
        v *= nodes[u].value
    node.value = v
    return node.value.shape


def backward_tnh(node, nodes):
    xidx = node.args[0]
    x = nodes[xidx].value
    grad = node.grad * (1 - np.tanh(x) ** 2)
    if nodes[xidx].grad is None:
        nodes[xidx].grad = grad
    else:
        nodes[xidx].grad += grad

def backward_rlu(node, nodes):
    alpha, xidx = node.args
    x = nodes[xidx].value
    grad = node.grad * rlu_grad(x, alpha)
    if nodes[xidx].grad is None:
        nodes[xidx].grad = grad
    else:
        nodes[xidx].grad += grad

def backward_mul(node, nodes):
    a, b = node.args
    A = nodes[a].value
    B = nodes[b].value
    grad = node.grad
    grad_a = np.dot(grad, B.T)
    grad_b = np.dot(A.T, grad)
    if nodes[a].grad is None:
        nodes[a].grad = grad_a
    else:
        nodes[a].grad += grad_a
    if nodes[b].grad is None:
        nodes[b].grad = grad_b
    else:
        nodes[b].grad += grad_b

def backward_sum(node, nodes):
    us = node.args
    for u in us:
        if nodes[u].grad is None:
            nodes[u].grad = node.grad.copy()
        else:
            nodes[u].grad += node.grad

def backward_had(node, nodes):
    us = node.args
    if len(us) == 1:
        if nodes[us[0]].grad is None:
            nodes[us[0]].grad = node.grad.copy()
        else:
            nodes[us[0]].grad += node.grad
    else:
        for j, u in enumerate(us):
            prod = node.grad.copy() if hasattr(node.grad, 'copy') else np.copy(node.grad)
            for k, v in enumerate(us):
                if k != j:
                    prod = prod * nodes[v].value
            if nodes[u].grad is None:
                nodes[u].grad = prod
            else:
                nodes[u].grad += prod

def print_matrix(mat):
    for row in mat:
        print(' '.join(f'{x:.10f}' for x in row))

def main():
    inp = sys.stdin.read().splitlines()
    N, M, K = map(int, inp.pop(0).split())
    nodes = []
    shapes = []

    # Parse computation graph
    def add_node(node, shape=None):
        nodes.append(node)
        shapes.append(shape)

    op_parsers = {
        'var': lambda parts: (Node('var', (int(parts[1]), int(parts[2]))), (int(parts[1]), int(parts[2]))),
        'tnh': lambda parts: (Node('tnh', (int(parts[1]) - 1,)), None),
        'rlu': lambda parts: (Node('rlu', (1.0 / int(parts[1]), int(parts[2]) - 1)), None),
        'mul': lambda parts: (Node('mul', (int(parts[1]) - 1, int(parts[2]) - 1)), None),
        'sum': lambda parts: (Node('sum', tuple(int(x) - 1 for x in parts[2:2 + int(parts[1])])), None),
        'had': lambda parts: (Node('had', tuple(int(x) - 1 for x in parts[2:2 + int(parts[1])])), None),
    }
    for _ in range(N):
        parts = inp.pop(0).split()
        node, shape = op_parsers[parts[0]](parts)
        add_node(node, shape)

    # Read input matrices for variables
    for i in range(M):
        r, c = nodes[i].args
        nodes[i].value = read_matrix(r, c, inp)
        shapes[i] = (r, c)

    # Forward pass
    forward_dispatch = {
        'tnh': forward_tnh,
        'rlu': forward_rlu,
        'mul': forward_mul,
        'sum': forward_sum,
        'had': forward_had,
    }
    for i in range(M, N):
        node = nodes[i]
        if node.op in forward_dispatch:
            shapes[i] = forward_dispatch[node.op](node, nodes)

    # Read output gradients
    outs = [N - K + i for i in range(K)]
    for idx in outs:
        r, c = shapes[idx]
        nodes[idx].grad = read_matrix(r, c, inp)

    # Backward pass
    backward_dispatch = {
        'tnh': backward_tnh,
        'rlu': backward_rlu,
        'mul': backward_mul,
        'sum': backward_sum,
        'had': backward_had,
    }
    for i in range(N - 1, -1, -1):
        node = nodes[i]
        if node.grad is None or node.op == 'var':
            continue
        if node.op in backward_dispatch:
            backward_dispatch[node.op](node, nodes)

    # Print outputs
    for idx in outs:
        print_matrix(nodes[idx].value)
    for i in range(M):
        if nodes[i].grad is None:
            r, c = shapes[i]
            nodes[i].grad = np.zeros((r, c))
        print_matrix(nodes[i].grad)

if __name__ == '__main__':
    main()