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
public class RobertaIntermediate : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    private readonly nn.Module<Tensor, Tensor> intermediate_act_fn;

    public RobertaIntermediate(RobertaConfig config) : base(nameof(RobertaIntermediate))
    {
        this.dense = nn.Linear(config.hidden_size, config.intermediate_size);
        this.intermediate_act_fn = config.hidden_act is nn.Module<Tensor, Tensor> m ? m : (config.hidden_act?.ToString() ?? "gelu").GetActivationFunction();
    }

    public override Tensor forward(Tensor hidden_states)
    {
        hidden_states = this.dense.call(hidden_states);
        hidden_states = this.intermediate_act_fn.call(hidden_states);
        return hidden_states;
    }
}
