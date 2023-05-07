// Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright (c) 2023, Andrew Chisholm
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace NanoGPTSharp;

using System.Diagnostics.Contracts;
using System.Collections.Generic;

public readonly record struct RobertaLayerArgs(Tensor? attention_mask = null, Tensor? head_mask = null, Tensor? encoder_hidden_states = null, Tensor? encoder_attention_mask = null, List<Tensor>? past_key_value = null, bool output_attentions = false);

public class RobertaLayer : nn.Module<Tensor, RobertaLayerArgs, IReadOnlyList<Tensor>>
{
    private RobertaConfig config;
    private readonly int chunk_size_feed_forward;
    private readonly int seq_len_dim;
    private readonly RobertaAttention attention;
    private readonly bool is_decoder;
    private readonly bool add_cross_attention;
    private readonly RobertaAttention crossattention;
    private readonly RobertaIntermediate intermediate;
    private readonly RobertaOutput output;

    public RobertaLayer(RobertaConfig config) : base(nameof(RobertaLayer))
    {
        this.config = config;
        this.chunk_size_feed_forward = config.chunk_size_feed_forward;
        this.seq_len_dim = 1;
        this.attention = new RobertaAttention(config);
        this.is_decoder = config.is_decoder is true;
        this.add_cross_attention = config.add_cross_attention is true;
        if (this.add_cross_attention) {
            if (!this.is_decoder)
            {
                throw new InvalidOperationException($"{this} should be used as a decoder model if cross attention is added");
            }

            this.crossattention = new RobertaAttention(config, position_embedding_type: "absolute");
        }
        this.intermediate = new RobertaIntermediate(config);
        this.output = new RobertaOutput(config);
    }

    public override IReadOnlyList<Tensor> forward(Tensor hidden_states, RobertaLayerArgs optional)
    {
        var (attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions) = optional;
        
        // decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        var self_attn_past_key_value = past_key_value is not null ? past_key_value.Slice(..2).ToList() : null;
        var self_attention_outputs = this.attention.call(
            hidden_states,
            new(attention_mask,
            head_mask,
            output_attentions: output_attentions,
            past_key_value: self_attn_past_key_value));
        var attention_output = self_attention_outputs[0];
        
        // if decoder, the last output is tuple of self-attn cache
        IReadOnlyList<Tensor> outputs;
        Tensor? present_key_value = null;
        if (this.is_decoder)
        {
            outputs = self_attention_outputs.Slice(1..-1);
            present_key_value = self_attention_outputs[-1];
        }
        else
        {
            outputs = self_attention_outputs.Slice(1..);  // add self attentions if we output attention weights
        }
        
        Tensor cross_attn_present_key_value;
        if (this.is_decoder && encoder_hidden_states is not null && present_key_value is not null)
        {
            Contract.Assert(this.crossattention is not null, $"If `encoder_hidden_states` are passed, {this} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`");
            var cross_attn_past_key_value = past_key_value is not null ? past_key_value.Slice(-2..) : null;
            // cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            var cross_attention_outputs = this.crossattention.call(
                attention_output,
                new(attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_past_key_value,
                    output_attentions));
            
            attention_output = cross_attention_outputs[0];
                        
            outputs = outputs.Concat(cross_attention_outputs.Slice(1..-1)).ToList();  // add cross attentions if we output attention weights

            // add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1];
            present_key_value = present_key_value + cross_attn_present_key_value;
        }
        
        var layer_output = attention_output.apply_chunking_to_forward(this.feed_forward_chunk, this.chunk_size_feed_forward, this.seq_len_dim);
        outputs = new[] { layer_output }.Concat(outputs).ToList();
            
        // if decoder, return the attn key/values as the last output
        if (this.is_decoder) {
            outputs = outputs.Append(present_key_value).ToList();
        }

        return outputs;
    }

    private Tensor feed_forward_chunk(Tensor attention_output)
    {
        var intermediate_output = this.intermediate.call(attention_output);
        var layer_output = this.output.call(intermediate_output, attention_output);
        return layer_output;
    }
}
