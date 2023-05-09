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
using System.Linq;
using TorchSharp.Modules;

public class RobertaEmbeddings : nn.Module<Tensor, Tensor, Tensor, Tensor, int, Tensor>
{
    private readonly RobertaConfig config;
    public readonly Embedding word_embeddings;
    public readonly Embedding position_embeddings;
    public readonly Embedding token_type_embeddings;

    public readonly TorchSharp.Modules.LayerNorm LayerNorm;

    public readonly Dropout dropout;
    public readonly string position_embedding_type;
    public readonly int padding_idx;

    internal RobertaEmbeddings(RobertaConfig config) : base(nameof(RobertaEmbeddings))
    {
        Contract.Assert(config.vocab_size is not 0);
        this.config = config;

        // Word Token Embeddings
        this.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx: config.pad_token_id);

        // Word Positional Embeddings
        this.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size);

        // Embedding to distinguish between two sentences.
        this.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size);

        this.LayerNorm = nn.LayerNorm(config.hidden_size, eps: config.layer_norm_eps);

        this.dropout = nn.Dropout(config.hidden_dropout_prob);

        this.position_embedding_type = config.position_embedding_type;

        this.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)));
        this.register_buffer("token_type_ids", torch.zeros(this.position_ids.size(), dtype: torch.@long)); // TODO: persistent: false not available with TorchSharp?
        
        this.padding_idx = config.pad_token_id;

        this.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx: this.padding_idx);
        this.RegisterComponents();
    }

    public Tensor position_ids => this.get_buffer("position_ids");

    public Tensor token_type_ids => this.get_buffer("token_type_ids");

    public override Tensor forward(Tensor? input_ids = null, Tensor? token_type_ids = null, Tensor? position_ids = null, Tensor? inputs_embeds = null, int past_key_values_length = 0)
    {
        if (position_ids is null)
        {
            if (input_ids is not null)
            {
                // Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = this.create_position_ids_from_input_ids(input_ids, this.padding_idx, past_key_values_length);
            }
            else
            {
                position_ids = this.create_position_ids_from_inputs_embeds(inputs_embeds);
            }
        }

        var input_shape = input_ids is not null ? input_ids.size() : inputs_embeds.size()[0.. -1].ToArray();
        var seq_length = input_shape[1];

        // Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        // when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids,
        if (token_type_ids is null)
        {
            if (has_buffer("token_type_ids"))
            {
                var buffered_token_type_ids = get_buffer("token_type_ids")[.., ..(int)seq_length];
                var buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length);
                token_type_ids = buffered_token_type_ids_expanded;
            }
            else
            {
                token_type_ids = torch.zeros(input_shape, dtype: torch.@long, device: this.position_ids.device);
            }
        }

        inputs_embeds ??= this.word_embeddings.forward(input_ids);
        
        var token_type_embeddings = this.token_type_embeddings.forward(token_type_ids);
        var embeddings = inputs_embeds + token_type_embeddings;

        if (this.position_embedding_type == "absolute")
        {
            var position_embeddings = this.position_embeddings.forward(position_ids);
            embeddings += position_embeddings;
        }

        embeddings = this.LayerNorm.forward(embeddings);
        embeddings = this.dropout.forward(embeddings);

        return embeddings;
    }

    public Tensor create_position_ids_from_inputs_embeds(Tensor inputsEmbeds)
    {
        var inputShape = inputsEmbeds.size()[0..-1].ToArray();
        var sequenceLength = inputShape[1];

        Tensor positionIds = torch.arange(this.padding_idx + 1, sequenceLength + this.padding_idx + 1, dtype: torch.int64, device: inputsEmbeds.device);
        return positionIds.unsqueeze(0).expand(inputShape);
    }

    /// <summary>
    /// Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    /// are ignored. This is modified from fairseq's `utils.make_positions`.
    /// </summary>
    /// <param name="input_ids"></param>
    /// <param name="padding_idx"></param>
    /// <param name="past_key_values_length"></param>
    /// <returns></returns>
    public Tensor create_position_ids_from_input_ids(Tensor input_ids, Tensor padding_idx, int past_key_values_length = 0)
    {
        // The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        var mask = input_ids.ne(padding_idx).@int();
        var incremental_indices = (torch.cumsum(mask, dim: 1).type_as(mask) + past_key_values_length) * mask;
        return incremental_indices.@long() + padding_idx;
    }
}
