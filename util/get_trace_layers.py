
import util.nethook as nethook

def get_trace_layers(model, hparams):
    try:
        n_layers = model.config.n_layer
    except:
        n_layers = model.config.num_hidden_layers


    lm_head, ln_f = (
        nethook.get_module(model, hparams.lm_head_module),
        nethook.get_module(model, hparams.ln_f_module),
    )

    #get encoder layers
    trace_layers = [
        hparams.layer_module_tmp.format(layer) for layer in range(n_layers)
    ]
    #Add final layer layernorm to trace layers
    #trace_layers.append(hparams.ln_f_module)

    #get layernorm layers
    ln_1_layers, ln_2_layers = {}, {}
    for layer_num in range(n_layers):
        ln_1_layers[layer_num] =  nethook.get_module(model, hparams.ln_1_module.format(layer_num))
        ln_2_layers[layer_num] =  nethook.get_module(model, hparams.ln_2_module.format(layer_num))
    
    return trace_layers, ln_1_layers, ln_2_layers, lm_head, ln_f, n_layers





def get_trace_layers_memories(model, hparams):
    try:
        n_layers = model.config.n_layer
    except:
        n_layers = model.config.num_hidden_layers


    #get encoder layers
    trace_layers = [
        hparams.rewrite_module_tmp.format(layer) for layer in range(n_layers)
    ]

    
    return trace_layers, n_layers