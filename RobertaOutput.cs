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
public class RobertaOutput : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly RobertaConfig config;
    private readonly Linear dense;
    private readonly TorchSharp.Modules.LayerNorm LayerNorm;
    private readonly Dropout dropout;

    public RobertaOutput(RobertaConfig config) : base(nameof(RobertaOutput))
    {
        this.config = config;
        this.dense = nn.Linear(config.intermediate_size, config.hidden_size);
        this.LayerNorm = nn.LayerNorm(config.hidden_size, eps:config.layer_norm_eps);
        this.dropout = nn.Dropout(config.hidden_dropout_prob);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor hidden_states, Tensor input_tensor)
    {
        hidden_states = this.dense.call(hidden_states);
        hidden_states = this.dropout.call(hidden_states);
        hidden_states = this.LayerNorm.call(hidden_states + input_tensor);
        return hidden_states;
    }
}
