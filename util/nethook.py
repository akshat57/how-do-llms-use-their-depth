"""
Utilities for instrumenting a torch model.

Trace will hook one layer at a time.
TraceDict will hook multiple layers at once.
subsequence slices intervals from Sequential modules.
get_module, replace_module, get_parameter resolve dotted names.
set_requires_grad recursively sets requires_grad in module parameters.
"""

import contextlib
import copy
import inspect
from collections import OrderedDict

import torch


class Trace(contextlib.AbstractContextManager):
    """
    To retain the output of the named layer during the computation of
    the given network:

        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a direct reference to
    the output object is returned, but options can control this:

        clone=True  - retains a copy of the output, which can be
            useful if you want to see the output before it might
            be modified by the network in-place later.
        detach=True - retains a detached reference or copy.  (By
            default the value would be left attached to the graph.)
        retain_grad=True - request gradient to be retained on the
            output.  After backward(), ret.output.grad is populated.

        retain_input=True - also retains the input.
        retain_output=False - can disable retaining the output.
        edit_output=fn - calls the function to modify the output
            of the layer before passing it the rest of the model.
            fn can optionally accept (output, layer) arguments
            for the original output and the layer name.
        stop=True - throws a StopForward exception after the layer
            is run, which allows running just a portion of a model.
    """

    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            module = get_module(module, layer)

        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_grad applies to output only.
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop

        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."


def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):
    """
    Creates a subsequence of a pytorch Sequential model, copying over
    modules together with parameters for the subsequence.  Only
    modules from first_layer to last_layer (inclusive) are included,
    or modules between after_layer and upto_layer (exclusive).
    Handles descent into dotted layer names as long as all references
    are within nested Sequential models.

    If share_weights is True, then references the original modules
    and their parameters without copying them.  Otherwise, by default,
    makes a separate brand-new copy.
    """
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )


def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    Recursive helper for subsequence() to support descent into dotted
    layer names.  In this helper, first, last, after, and upto are
    arrays of names resulting from splitting on dots.  Can only
    descend into nested Sequentials.
    """
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    # A = current level short name of A.
    # AN = full name for recursive descent if not innermost.
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:  # just like F if not a leaf.
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            # AR = full name for recursive descent if name matches.
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # just like L if not a leaf.
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # Omit empty subsequences except at the outermost level,
    # where we should not return None.
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    # print([i[0] for i in model.named_modules()])
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)

['', 'model', 'model.transformer', 'model.transformer.wte', 'model.transformer.emb_drop', 'model.transformer.ln_f', 'model.transformer.blocks', 'model.transformer.blocks.0', 'model.transformer.blocks.0.dropout', 'model.transformer.blocks.0.act', 'model.transformer.blocks.0.attn_out', 'model.transformer.blocks.0.ff_out', 'model.transformer.blocks.0.rotary_emb', 'model.transformer.blocks.0.att_proj', 'model.transformer.blocks.0.ff_proj', 'model.transformer.blocks.0.attn_norm', 'model.transformer.blocks.0.ff_norm', 'model.transformer.blocks.1', 'model.transformer.blocks.1.dropout', 'model.transformer.blocks.1.act', 'model.transformer.blocks.1.attn_out', 'model.transformer.blocks.1.ff_out', 'model.transformer.blocks.1.rotary_emb', 'model.transformer.blocks.1.att_proj', 'model.transformer.blocks.1.ff_proj', 'model.transformer.blocks.1.attn_norm', 'model.transformer.blocks.1.ff_norm', 'model.transformer.blocks.2', 'model.transformer.blocks.2.dropout', 'model.transformer.blocks.2.act', 'model.transformer.blocks.2.attn_out', 'model.transformer.blocks.2.ff_out', 'model.transformer.blocks.2.rotary_emb', 'model.transformer.blocks.2.att_proj', 'model.transformer.blocks.2.ff_proj', 'model.transformer.blocks.2.attn_norm', 'model.transformer.blocks.2.ff_norm', 'model.transformer.blocks.3', 'model.transformer.blocks.3.dropout', 'model.transformer.blocks.3.act', 'model.transformer.blocks.3.attn_out', 'model.transformer.blocks.3.ff_out', 'model.transformer.blocks.3.rotary_emb', 'model.transformer.blocks.3.att_proj', 'model.transformer.blocks.3.ff_proj', 'model.transformer.blocks.3.attn_norm', 'model.transformer.blocks.3.ff_norm', 'model.transformer.blocks.4', 'model.transformer.blocks.4.dropout', 'model.transformer.blocks.4.act', 'model.transformer.blocks.4.attn_out', 'model.transformer.blocks.4.ff_out', 'model.transformer.blocks.4.rotary_emb', 'model.transformer.blocks.4.att_proj', 'model.transformer.blocks.4.ff_proj', 'model.transformer.blocks.4.attn_norm', 'model.transformer.blocks.4.ff_norm', 'model.transformer.blocks.5', 'model.transformer.blocks.5.dropout', 'model.transformer.blocks.5.act', 'model.transformer.blocks.5.attn_out', 'model.transformer.blocks.5.ff_out', 'model.transformer.blocks.5.rotary_emb', 'model.transformer.blocks.5.att_proj', 'model.transformer.blocks.5.ff_proj', 'model.transformer.blocks.5.attn_norm', 'model.transformer.blocks.5.ff_norm', 'model.transformer.blocks.6', 'model.transformer.blocks.6.dropout', 'model.transformer.blocks.6.act', 'model.transformer.blocks.6.attn_out', 'model.transformer.blocks.6.ff_out', 'model.transformer.blocks.6.rotary_emb', 'model.transformer.blocks.6.att_proj', 'model.transformer.blocks.6.ff_proj', 'model.transformer.blocks.6.attn_norm', 'model.transformer.blocks.6.ff_norm', 'model.transformer.blocks.7', 'model.transformer.blocks.7.dropout', 'model.transformer.blocks.7.act', 'model.transformer.blocks.7.attn_out', 'model.transformer.blocks.7.ff_out', 'model.transformer.blocks.7.rotary_emb', 'model.transformer.blocks.7.att_proj', 'model.transformer.blocks.7.ff_proj', 'model.transformer.blocks.7.attn_norm', 'model.transformer.blocks.7.ff_norm', 'model.transformer.blocks.8', 'model.transformer.blocks.8.dropout', 'model.transformer.blocks.8.act', 'model.transformer.blocks.8.attn_out', 'model.transformer.blocks.8.ff_out', 'model.transformer.blocks.8.rotary_emb', 'model.transformer.blocks.8.att_proj', 'model.transformer.blocks.8.ff_proj', 'model.transformer.blocks.8.attn_norm', 'model.transformer.blocks.8.ff_norm', 'model.transformer.blocks.9', 'model.transformer.blocks.9.dropout', 'model.transformer.blocks.9.act', 'model.transformer.blocks.9.attn_out', 'model.transformer.blocks.9.ff_out', 'model.transformer.blocks.9.rotary_emb', 'model.transformer.blocks.9.att_proj', 'model.transformer.blocks.9.ff_proj', 'model.transformer.blocks.9.attn_norm', 'model.transformer.blocks.9.ff_norm', 'model.transformer.blocks.10', 'model.transformer.blocks.10.dropout', 'model.transformer.blocks.10.act', 'model.transformer.blocks.10.attn_out', 'model.transformer.blocks.10.ff_out', 'model.transformer.blocks.10.rotary_emb', 'model.transformer.blocks.10.att_proj', 'model.transformer.blocks.10.ff_proj', 'model.transformer.blocks.10.attn_norm', 'model.transformer.blocks.10.ff_norm', 'model.transformer.blocks.11', 'model.transformer.blocks.11.dropout', 'model.transformer.blocks.11.act', 'model.transformer.blocks.11.attn_out', 'model.transformer.blocks.11.ff_out', 'model.transformer.blocks.11.rotary_emb', 'model.transformer.blocks.11.att_proj', 'model.transformer.blocks.11.ff_proj', 'model.transformer.blocks.11.attn_norm', 'model.transformer.blocks.11.ff_norm', 'model.transformer.blocks.12', 'model.transformer.blocks.12.dropout', 'model.transformer.blocks.12.act', 'model.transformer.blocks.12.attn_out', 'model.transformer.blocks.12.ff_out', 'model.transformer.blocks.12.rotary_emb', 'model.transformer.blocks.12.att_proj', 'model.transformer.blocks.12.ff_proj', 'model.transformer.blocks.12.attn_norm', 'model.transformer.blocks.12.ff_norm', 'model.transformer.blocks.13', 'model.transformer.blocks.13.dropout', 'model.transformer.blocks.13.act', 'model.transformer.blocks.13.attn_out', 'model.transformer.blocks.13.ff_out', 'model.transformer.blocks.13.rotary_emb', 'model.transformer.blocks.13.att_proj', 'model.transformer.blocks.13.ff_proj', 'model.transformer.blocks.13.attn_norm', 'model.transformer.blocks.13.ff_norm', 'model.transformer.blocks.14', 'model.transformer.blocks.14.dropout', 'model.transformer.blocks.14.act', 'model.transformer.blocks.14.attn_out', 'model.transformer.blocks.14.ff_out', 'model.transformer.blocks.14.rotary_emb', 'model.transformer.blocks.14.att_proj', 'model.transformer.blocks.14.ff_proj', 'model.transformer.blocks.14.attn_norm', 'model.transformer.blocks.14.ff_norm', 'model.transformer.blocks.15', 'model.transformer.blocks.15.dropout', 'model.transformer.blocks.15.act', 'model.transformer.blocks.15.attn_out', 'model.transformer.blocks.15.ff_out', 'model.transformer.blocks.15.rotary_emb', 'model.transformer.blocks.15.att_proj', 'model.transformer.blocks.15.ff_proj', 'model.transformer.blocks.15.attn_norm', 'model.transformer.blocks.15.ff_norm', 'model.transformer.blocks.16', 'model.transformer.blocks.16.dropout', 'model.transformer.blocks.16.act', 'model.transformer.blocks.16.attn_out', 'model.transformer.blocks.16.ff_out', 'model.transformer.blocks.16.rotary_emb', 'model.transformer.blocks.16.att_proj', 'model.transformer.blocks.16.ff_proj', 'model.transformer.blocks.16.attn_norm', 'model.transformer.blocks.16.ff_norm', 'model.transformer.blocks.17', 'model.transformer.blocks.17.dropout', 'model.transformer.blocks.17.act', 'model.transformer.blocks.17.attn_out', 'model.transformer.blocks.17.ff_out', 'model.transformer.blocks.17.rotary_emb', 'model.transformer.blocks.17.att_proj', 'model.transformer.blocks.17.ff_proj', 'model.transformer.blocks.17.attn_norm', 'model.transformer.blocks.17.ff_norm', 'model.transformer.blocks.18', 'model.transformer.blocks.18.dropout', 'model.transformer.blocks.18.act', 'model.transformer.blocks.18.attn_out', 'model.transformer.blocks.18.ff_out', 'model.transformer.blocks.18.rotary_emb', 'model.transformer.blocks.18.att_proj', 'model.transformer.blocks.18.ff_proj', 'model.transformer.blocks.18.attn_norm', 'model.transformer.blocks.18.ff_norm', 'model.transformer.blocks.19', 'model.transformer.blocks.19.dropout', 'model.transformer.blocks.19.act', 'model.transformer.blocks.19.attn_out', 'model.transformer.blocks.19.ff_out', 'model.transformer.blocks.19.rotary_emb', 'model.transformer.blocks.19.att_proj', 'model.transformer.blocks.19.ff_proj', 'model.transformer.blocks.19.attn_norm', 'model.transformer.blocks.19.ff_norm', 'model.transformer.blocks.20', 'model.transformer.blocks.20.dropout', 'model.transformer.blocks.20.act', 'model.transformer.blocks.20.attn_out', 'model.transformer.blocks.20.ff_out', 'model.transformer.blocks.20.rotary_emb', 'model.transformer.blocks.20.att_proj', 'model.transformer.blocks.20.ff_proj', 'model.transformer.blocks.20.attn_norm', 'model.transformer.blocks.20.ff_norm', 'model.transformer.blocks.21', 'model.transformer.blocks.21.dropout', 'model.transformer.blocks.21.act', 'model.transformer.blocks.21.attn_out', 'model.transformer.blocks.21.ff_out', 'model.transformer.blocks.21.rotary_emb', 'model.transformer.blocks.21.att_proj', 'model.transformer.blocks.21.ff_proj', 'model.transformer.blocks.21.attn_norm', 'model.transformer.blocks.21.ff_norm', 'model.transformer.blocks.22', 'model.transformer.blocks.22.dropout', 'model.transformer.blocks.22.act', 'model.transformer.blocks.22.attn_out', 'model.transformer.blocks.22.ff_out', 'model.transformer.blocks.22.rotary_emb', 'model.transformer.blocks.22.att_proj', 'model.transformer.blocks.22.ff_proj', 'model.transformer.blocks.22.attn_norm', 'model.transformer.blocks.22.ff_norm', 'model.transformer.blocks.23', 'model.transformer.blocks.23.dropout', 'model.transformer.blocks.23.act', 'model.transformer.blocks.23.attn_out', 'model.transformer.blocks.23.ff_out', 'model.transformer.blocks.23.rotary_emb', 'model.transformer.blocks.23.att_proj', 'model.transformer.blocks.23.ff_proj', 'model.transformer.blocks.23.attn_norm', 'model.transformer.blocks.23.ff_norm', 'model.transformer.blocks.24', 'model.transformer.blocks.24.dropout', 'model.transformer.blocks.24.act', 'model.transformer.blocks.24.attn_out', 'model.transformer.blocks.24.ff_out', 'model.transformer.blocks.24.rotary_emb', 'model.transformer.blocks.24.att_proj', 'model.transformer.blocks.24.ff_proj', 'model.transformer.blocks.24.attn_norm', 'model.transformer.blocks.24.ff_norm', 'model.transformer.blocks.25', 'model.transformer.blocks.25.dropout', 'model.transformer.blocks.25.act', 'model.transformer.blocks.25.attn_out', 'model.transformer.blocks.25.ff_out', 'model.transformer.blocks.25.rotary_emb', 'model.transformer.blocks.25.att_proj', 'model.transformer.blocks.25.ff_proj', 'model.transformer.blocks.25.attn_norm', 'model.transformer.blocks.25.ff_norm', 'model.transformer.blocks.26', 'model.transformer.blocks.26.dropout', 'model.transformer.blocks.26.act', 'model.transformer.blocks.26.attn_out', 'model.transformer.blocks.26.ff_out', 'model.transformer.blocks.26.rotary_emb', 'model.transformer.blocks.26.att_proj', 'model.transformer.blocks.26.ff_proj', 'model.transformer.blocks.26.attn_norm', 'model.transformer.blocks.26.ff_norm', 'model.transformer.blocks.27', 'model.transformer.blocks.27.dropout', 'model.transformer.blocks.27.act', 'model.transformer.blocks.27.attn_out', 'model.transformer.blocks.27.ff_out', 'model.transformer.blocks.27.rotary_emb', 'model.transformer.blocks.27.att_proj', 'model.transformer.blocks.27.ff_proj', 'model.transformer.blocks.27.attn_norm', 'model.transformer.blocks.27.ff_norm', 'model.transformer.blocks.28', 'model.transformer.blocks.28.dropout', 'model.transformer.blocks.28.act', 'model.transformer.blocks.28.attn_out', 'model.transformer.blocks.28.ff_out', 'model.transformer.blocks.28.rotary_emb', 'model.transformer.blocks.28.att_proj', 'model.transformer.blocks.28.ff_proj', 'model.transformer.blocks.28.attn_norm', 'model.transformer.blocks.28.ff_norm', 'model.transformer.blocks.29', 'model.transformer.blocks.29.dropout', 'model.transformer.blocks.29.act', 'model.transformer.blocks.29.attn_out', 'model.transformer.blocks.29.ff_out', 'model.transformer.blocks.29.rotary_emb', 'model.transformer.blocks.29.att_proj', 'model.transformer.blocks.29.ff_proj', 'model.transformer.blocks.29.attn_norm', 'model.transformer.blocks.29.ff_norm', 'model.transformer.blocks.30', 'model.transformer.blocks.30.dropout', 'model.transformer.blocks.30.act', 'model.transformer.blocks.30.attn_out', 'model.transformer.blocks.30.ff_out', 'model.transformer.blocks.30.rotary_emb', 'model.transformer.blocks.30.att_proj', 'model.transformer.blocks.30.ff_proj', 'model.transformer.blocks.30.attn_norm', 'model.transformer.blocks.30.ff_norm', 'model.transformer.blocks.31', 'model.transformer.blocks.31.dropout', 'model.transformer.blocks.31.act', 'model.transformer.blocks.31.attn_out', 'model.transformer.blocks.31.ff_out', 'model.transformer.blocks.31.rotary_emb', 'model.transformer.blocks.31.att_proj', 'model.transformer.blocks.31.ff_proj', 'model.transformer.blocks.31.attn_norm', 'model.transformer.blocks.31.ff_norm', 'model.transformer.ff_out']

def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    (1) arguments with matching names are passed by name.
    (2) remaining non-name-matched args are passed by order.
    (3) extra caller arguments that the function cannot
        accept are not passed.
    (4) extra required function arguments that the caller
        cannot provide cause a TypeError to be raised.
    Ordinary python calling conventions are helpful for
    supporting a function that might be revised to accept
    extra arguments in a newer version, without requiring the
    caller to pass those new arguments.  This function helps
    support function callers that might be revised to supply
    extra arguments, without requiring the callee to accept
    those new arguments.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # Pass positional args that match name first, then by position.
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # Fill unmatched positional args with unmatched keyword args in order.
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # Pass remaining kw args if they can be accepted.
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # Pass remaining positional args if they can be accepted.
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)
