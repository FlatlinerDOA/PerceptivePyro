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

public readonly record struct RobertaAttentionArgs(Tensor? attention_mask = null, Tensor? head_mask = null, Tensor? encoder_hidden_states = null, Tensor? encoder_attention_mask = null, IReadOnlyList<Tensor>? past_key_value = null, bool output_attentions = false);

public class RobertaAttention : nn.Module<Tensor, RobertaAttentionArgs, IReadOnlyList<Tensor>>
{
    private readonly RobertaConfig config;
    private readonly RobertaSelfAttention self;
    private readonly RobertaSelfOutput output;
    private readonly HashSet<int> pruned_heads;

    public RobertaAttention(RobertaConfig config, string position_embedding_type = null) : base(nameof(RobertaAttention))
    {
        this.config = config;
        this.self = new RobertaSelfAttention(config, position_embedding_type: position_embedding_type);
        this.output = new RobertaSelfOutput(config);
        this.pruned_heads = new HashSet<int>();
    }

    /*
     //Head pruning not included for now as the model is immutable once constructed. Should be constructed pruned.
    public void prune_heads(List<int> heads)
    {
        if (heads.Count == 0)
        {
            return;
        }

        (heads, var index) = find_pruneable_heads_and_indices(heads, this.self.num_attention_heads, this.self.attention_head_size, this.pruned_heads);
        // Prune linear layers
        this.self.query = prune_linear_layer(this.self.query, index);
        this.self.key = prune_linear_layer(this.self.key, index);
        this.self.value = prune_linear_layer(this.self.value, index);
        this.output.dense = prune_linear_layer(this.output.dense, index, dim:1);

        // Update hyper params and store pruned heads
        this.self.num_attention_heads = this.self.num_attention_heads - heads.Count;
        this.self.all_head_size = this.self.attention_head_size * this.self.num_attention_heads;
        this.pruned_heads.UnionWith(heads);
    }*/

    public override IReadOnlyList<Tensor> forward(Tensor hidden_states, RobertaAttentionArgs optional)
    {
        var self_outputs = this.self.call(
            hidden_states,
            new(optional.attention_mask,
            optional.head_mask,
            optional.encoder_hidden_states,
            optional.encoder_attention_mask,
            optional.past_key_value,
            optional.output_attentions));
        var attention_output = this.output.call(self_outputs[0], hidden_states);
        var outputs = new[] { attention_output }.Concat(self_outputs.Skip(1)).ToList();  // add attentions if we output them
        return outputs;
    }
}
