import warnings

from .handlers import handlers
from .utils.trace import trace

__all__ = ['profile_model']


def profile_model(model, args=(), kwargs=None, reduction=sum, verbose=False):
    results = dict()
    peak_memory = dict()
    graph = trace(model, args, kwargs)
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    results[node], peak_memory[node] = func(node, verbose)
                break
        else:
            warnings.warn('No handlers found: "{}". Skipped.'.format(
                node.operator))

    if reduction is not None:
        return reduction(results.values()), max(peak_memory.values())
    else:
        return results, peak_memory
