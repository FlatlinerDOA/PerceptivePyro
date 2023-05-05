namespace NanoGPTSharp;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TorchSharp.Modules;

public readonly record struct RobertaEncoderArgs(
    Tensor? attention_mask = null,
    Tensor? head_mask = null,
    Tensor? encoder_hidden_states = null,
    Tensor? encoder_attention_mask = null,
    IReadOnlyList<Tensor>? past_key_values = null,
    bool use_cache = false,
    bool output_attentions = false,
    bool output_hidden_states = false);

public class RobertaEncoder : nn.Module<Tensor, RobertaEncoderArgs, BaseModelOutputWithPastAndCrossAttentions>
{
    private RobertaConfig config;
    private ModuleList<RobertaLayer> layer;
    private bool gradient_checkpointing;

    public RobertaEncoder(RobertaConfig config) : base(nameof(RobertaEncoder))
    {
        this.config = config;
        this.layer = nn.ModuleList((from _ in Enumerable.Range(0, this.config.num_hidden_layers) select new RobertaLayer(config)).ToArray());
        this.gradient_checkpointing = false;
        this.RegisterComponents();
    }

    // Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]
    public override BaseModelOutputWithPastAndCrossAttentions forward(Tensor hidden_states, RobertaEncoderArgs options)
    {
        var (attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states) = options;
        var all_hidden_states = output_hidden_states is true ? new List<Tensor>() : null;
        var all_self_attentions = output_attentions is true ? new List<Tensor>() : null;
        var all_cross_attentions = output_attentions is true && this.config.add_cross_attention is true ? new List<Tensor>() : null;
        if (this.gradient_checkpointing && this.training && use_cache is true)
        {
            Trace.TraceWarning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...");
            use_cache = false;
        }

        var next_decoder_cache = use_cache ? new List<Tensor>() : null;
        foreach (var (i, layer_module) in this.layer.Select((r, i) => (i, r)))
        {
            if (output_hidden_states is true)
            {
                all_hidden_states.Add(hidden_states);
            }

            var layer_head_mask = head_mask is not null ? head_mask[i] : null;
            var past_key_value = past_key_values is not null ? past_key_values[i] : null;
            IReadOnlyList<Tensor> layer_outputs;
            if (this.gradient_checkpointing && this.training)
            {
                throw new NotSupportedException("Gradient checkpointing not supported yet.");
                //var layer_outputs = torch.utils.data.checkpoint.checkpoint(
                //     (nn.Module module) => module.forward(layer_module, past_key_value, output_attentions),
                //     hidden_states,
                //     attention_mask,
                //     layer_head_mask,
                //     encoder_hidden_states,
                //     encoder_attention_mask);
            }
            else
            {
                layer_outputs = layer_module.forward(
                    hidden_states,
                    new(attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value is not null ? new List<Tensor> { past_key_value } : null,
                    output_attentions));
            }

            hidden_states = layer_outputs[0];
            if (use_cache is true)
            {
                next_decoder_cache.Add(layer_outputs[-1]);
            }

            if (output_attentions is true)
            {
                all_self_attentions.Add(layer_outputs[1]);
                if (this.config.add_cross_attention is true)
                {
                    all_cross_attentions.Add(layer_outputs[2]); 
                }
            }
        }

        if (output_hidden_states is true)
        {
            all_hidden_states.Add(hidden_states);
        }

        return new(hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attentions);  
    }
}
