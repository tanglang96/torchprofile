from .utils import math

__all__ = ['handlers']

def mul_all(s):
    if s == None:
        return 0
    r = 1
    for i in range(len(s)):
        r *= s[i]
    return r

def get_peak_memory(node):
    memory = 0
    input_size = 0
    output_size = 0
    n_inputs = len(node.inputs)
    n_outputs = len(node.outputs)
    for i in range(n_inputs):
        input_size += mul_all(node.inputs[i].shape)
    input_size = input_size * 4 / 1024
    for i in range(n_outputs):
        output_size += mul_all(node.outputs[i].shape)
    output_size = output_size * 4 / 1024
    memory = input_size + output_size
    return memory, input_size, output_size

def addmm(node, verbose):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    flops = n * m * p
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def addmv(node, verbose):
    # [n] = aten::addmv([n], [n, m], [m], *, *)
    n, m = node.inputs[1].shape
    flops = n * m
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def bmm(node, verbose):
    # [b, n, p] = aten::bmm([b, n, m], [b, m, p])
    b, n, m = node.inputs[0].shape
    b, m, p = node.inputs[1].shape
    flops = b * n * m * p
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def matmul(node, verbose):
    memory, input_size, output_size = get_peak_memory(node)
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        flops = n
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        flops = n * m
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        flops = n * m
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        flops = n * m * p
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        flops = math.prod(b) * n * m
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        flops = math.prod(b) * n * m
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        flops = math.prod(b) * n * m * p
        if memory > 0 and verbose:
            print('-'*70)
            print(node)
            print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
        return flops, memory


def mul(node, verbose):
    os = node.outputs[0].shape
    flops = math.prod(os)
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def convolution(node, verbose):
    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
        oc, ic, *ks = node.inputs[1].shape
    else:
        ic, oc, *ks = node.inputs[1].shape
    os = node.outputs[0].shape
    flops = math.prod(os) * ic * math.prod(ks)
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def norm(node, verbose):
    if node.operator in ['aten::batch_norm', 'aten::instance_norm']:
        affine = node.inputs[1].shape is not None
    elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
        affine = node.inputs[2].shape is not None
    else:
        raise ValueError(node.operator)

    os = node.outputs[0].shape
    flops = math.prod(os) if affine else 0
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def avg_pool_or_mean(node, verbose):
    os = node.outputs[0].shape
    flops = math.prod(os)
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def leaky_relu(node, verbose):
    os = node.outputs[0].shape
    flops = math.prod(os)
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory


def upsample_bilinear2d(node, verbose):
    os = node.outputs[0].shape
    flops = math.prod(os) * 4
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory

def others(node, verbose):
    flops = 0
    memory, input_size, output_size = get_peak_memory(node)
    if memory > 0 and verbose:
        print('-'*70)
        print(node.operator)
        print("FLOPs: %.2fm, memory: %.2fkB, input: %.2fkB, output: %.2fkB"%(flops/1e6, memory, input_size, output_size))
    return flops, memory

handlers = (
    ('aten::addmm', addmm),
    ('aten::addmv', addmv),
    ('aten::bmm', bmm),
    ('aten::matmul', matmul),
    (('aten::mul', 'aten::mul_'), mul),
    ('aten::_convolution', convolution),
    (('aten::batch_norm', 'aten::instance_norm', 'aten::layer_norm',
      'aten::group_norm'), norm),
    (('aten::adaptive_avg_pool1d', 'aten::adaptive_avg_pool2d',
      'aten::adaptive_avg_pool3d', 'aten::avg_pool1d', 'aten::avg_pool2d',
      'aten::avg_pool3d', 'aten::mean'), avg_pool_or_mean),
    ('aten::leaky_relu', leaky_relu),
    ('aten::upsample_bilinear2d', upsample_bilinear2d),
    (('aten::adaptive_max_pool1d', 'aten::adaptive_max_pool2d',
      'aten::adaptive_max_pool3d', 'aten::add', 'aten::add_',
      'aten::alpha_dropout', 'aten::cat', 'aten::chunk', 'aten::clamp',
      'aten::clone', 'aten::constant_pad_nd', 'aten::contiguous',
      'aten::detach', 'aten::div', 'aten::div_', 'aten::dropout',
      'aten::dropout_', 'aten::embedding', 'aten::eq', 'aten::feature_dropout',
      'aten::flatten', 'aten::floor', 'aten::floor_divide', 'aten::gt',
      'aten::hardtanh_', 'aten::index', 'aten::int', 'aten::log_softmax',
      'aten::lt', 'aten::max_pool1d', 'aten::max_pool1d_with_indices',
      'aten::max_pool2d', 'aten::max_pool2d_with_indices', 'aten::max_pool3d',
      'aten::max_pool3d_with_indices', 'aten::max_unpool1d',
      'aten::max_unpool2d', 'aten::max_unpool3d', 'aten::ne',
      'aten::reflection_pad1d', 'aten::reflection_pad2d',
      'aten::reflection_pad3d', 'aten::relu', 'aten::relu_',
      'aten::replication_pad1d', 'aten::replication_pad2d',
      'aten::replication_pad3d', 'aten::rsub', 'aten::select', 'aten::sigmoid',
      'aten::size', 'aten::slice', 'aten::softmax', 'aten::softshrink',
      'aten::squeeze', 'aten::stack', 'aten::sub', 'aten::sum', 'aten::t',
      'aten::tanh', 'aten::threshold', 'aten::to', 'aten::transpose',
      'aten::upsample_nearest2d', 'aten::view', 'aten::zeros',
      'prim::constant', 'prim::listconstruct', 'prim::listunpack',
      'prim::numtotensor', 'prim::tupleconstruct'), others),
)
