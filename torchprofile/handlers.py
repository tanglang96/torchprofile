import numpy as np

__all__ = ['handlers']


def addmm(node):
    # [n, p] = aten::addmm([n(, p)], [n, m], [m, p], *, *)
    n, p = node.outputs[0].shape
    n, m = node.inputs[1].shape
    return n * m * p


def addmv(node):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.outputs[1].shape
    return n * m


def bmm(node):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, p = node.outputs[0].shape
    b, n, m = node.inputs[0].shape
    return b * n * m * p


def matmul(node):
    return np.prod(node.inputs[0].shape + [node.inputs[1].shape[-1]])


def mul(node):
    os = node.outputs[0].shape
    return np.prod(os)


def convolution(node):
    os = node.outputs[0].shape
    gs, *ks = node.inputs[1].shape[1:]
    return np.prod(os) * np.prod(ks) * gs


def batch_norm(node):
    # TODO: provide an option to not fuse `batch_norm` into `linear` or `conv`
    return 0


def instance_norm_or_layer_norm(node):
    os = node.outputs[0].shape
    return np.prod(os)


def avg_pool_nd(node):
    os = node.outputs[0].shape
    return np.prod(os)


def mean(node):
    return 1


handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::matmul', matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    ('aten::batch_norm', batch_norm),
    (('aten::instance_norm', 'aten::layer_norm'), instance_norm_or_layer_norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d', 'aten::adaptive_avg_pool3d',
      'aten::avg_pool1d', 'aten::avg_pool2d', 'aten::avg_pool3d'), avg_pool_nd),
    ('aten::mean', mean),

    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d', 'aten::adaptive_max_pool3d', 'aten::add', 'aten::add_',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::div', 'aten::div_', 'aten::dropout', 'aten::dropout_', 'aten::embedding', 'aten::eq',
      'aten::feature_dropout', 'aten::flatten', 'aten::gt', 'aten::hardtanh_', 'aten::int', 'aten::lt',
      'aten::log_softmax', 'aten::max_pool1d', 'aten::max_pool1d_with_indices', 'aten::max_pool2d',
      'aten::max_pool2d_with_indices', 'aten::max_pool3d', 'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne', 'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::relu', 'aten::relu_', 'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::select', 'aten::sigmoid', 'aten::size', 'aten::slice', 'aten::softmax',
      'aten::softshrink', 'aten::sub', 'aten::sum', 'aten::t', 'aten::tanh', 'aten::threshold', 'aten::transpose',
      'aten::view', 'prim::constant', 'prim::listconstruct', 'prim::listunpack', 'prim::numtotensor'), None)
)
