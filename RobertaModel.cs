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
﻿using System.Diagnostics.Contracts;

namespace PerceptivePyro;

using F = TorchSharp.torch.nn.functional;

public record RobertaModelArgs(
    Tensor? input_ids = null,
    Tensor? attention_mask = null,
    Tensor? token_type_ids = null,
    Tensor? position_ids = null,
    Tensor? head_mask = null,
    Tensor? inputs_embeds = null,
    Tensor? encoder_hidden_states = null,
    Tensor? encoder_attention_mask = null,
    List<Tensor>? past_key_values = null,
    bool? use_cache = null,
    bool? output_attentions = null,
    bool? output_hidden_states = null);

/// <summary>
/// TorchLib Roberta Transformer (from HuggingFace).
/// Code ported from https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
/// </summary>
public class RobertaModel : nn.Module<RobertaModelArgs, BaseModelOutputWithPastAndCrossAttentions>
{
    private RobertaConfig config;
    private readonly RobertaPooler? pooler;
    private readonly RobertaEmbeddings embeddings;
    private readonly RobertaEncoder encoder;

    internal RobertaModel(RobertaConfig config, bool add_pooling_layer = true) : base(nameof(RobertaModel))
    {
        Contract.Assert(config.vocab_size is not 0);
        this.config = config;

        this.embeddings = new RobertaEmbeddings(config);
        this.encoder = new RobertaEncoder(config);

        this.pooler = add_pooling_layer ? new RobertaPooler(config) : null;

        // Initialize weights and apply final processing.
        this.post_init();

        this.RegisterComponents();
    }

    public RobertaConfig Config => this.config;
    
    public string? ModelName => this.Config._name_or_path;
    
    /// <summary>
    /// Gets the maximum number of input tokens prior to tokenization, this is the maximum position embedding size, minus the Beginning of Sentence (BOS) and End of Sentence (EOS) tokens.
    /// </summary>
    public int MaxInputTokenLength => this.config.max_position_embeddings - 2;
    
    public override BaseModelOutputWithPastAndCrossAttentions forward(RobertaModelArgs input)
    {
        var (
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states
            ) = input;
        output_attentions ??= this.config.output_attentions;
        output_hidden_states ??= this.config.output_hidden_states;
        if (this.config.is_decoder is true)
        {
            use_cache ??= this.config.use_cache;
        }
        else
        {
            use_cache = false;
        }

        Contract.Assert(input_ids is null != inputs_embeds is null, "You cannot specify both input_ids and inputs_embeds at the same time");

        long[] input_shape;
        if (input_ids is not null)
        {
            input_shape = input_ids.size();
        }
        else if (input_ids is not null)
        {
            input_shape = input_ids.size();
        }
        else if (inputs_embeds is not null)
        {
            input_shape = inputs_embeds.size()[..-1];
        }
        else
        {
            throw new InvalidOperationException("You have to specify either input_ids or inputs_embeds");
        }

        var (batch_size, seq_length) = (input_shape[0], input_shape[1]);
        var device = input_ids is not null ? input_ids.device : inputs_embeds.device;

        var past_key_values_length = past_key_values is not null ? past_key_values[0][0].shape[2] : 0;
        if (attention_mask is null)
        {
            attention_mask = torch.ones(new[] { batch_size, seq_length + past_key_values_length }, device: device);
        }

        if (token_type_ids is null)
        {
            if (this.embeddings.has_buffer("token_type_ids"))
            {
                var buffered_token_type_ids = this.embeddings.token_type_ids[.., ..(int)seq_length];
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length);
            }
            else
            {
                token_type_ids = torch.zeros(input_shape, dtype: torch.@long, device: device);
            }
        }

        // We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        // ourselves in which case we just need to make it broadcastable to all heads.
        var extended_attention_mask = this.get_extended_attention_mask(attention_mask, input_shape);

        // If a 2D or 3D attention mask is provided for the cross-attention
        // we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        Tensor? encoder_extended_attention_mask = null;
        if (this.config.is_decoder is true && encoder_hidden_states is not null)
        {
            var (encoder_batch_size, encoder_sequence_length) = (encoder_hidden_states.size()[0], encoder_hidden_states.size()[1]);
            var encoder_hidden_shape = new[] { encoder_batch_size, encoder_sequence_length };
            if (encoder_attention_mask is null)
            {
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device: device);
            }

            encoder_extended_attention_mask = this.invert_attention_mask(encoder_attention_mask);
        }

        // Prepare head mask if needed
        // 1.0 in head_mask indicate we keep the head
        // attention_probs has shape bsz x n_heads x N x N
        // input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        // and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = this.get_head_mask(head_mask, this.config.num_hidden_layers);

        var embedding_output = this.embeddings.call(
            input_ids,
            position_ids,
            token_type_ids,
            inputs_embeds,
            (int)past_key_values_length);

        var encoder_outputs = this.encoder.call(
            embedding_output,
            new(
                attention_mask: extended_attention_mask,
                head_mask: head_mask,
                encoder_hidden_states: encoder_hidden_states,
                encoder_attention_mask: encoder_extended_attention_mask,
                past_key_values: past_key_values,
                use_cache: use_cache ?? false,
                output_attentions: output_attentions ?? false,
                output_hidden_states: output_hidden_states ?? false));
        var sequence_output = encoder_outputs.last_hidden_state;
        var pooled_output = this.pooler is not null ? this.pooler.call(sequence_output) : null;
        return new BaseModelOutputWithPastAndCrossAttentions(
            sequence_output,
            pooled_output,
            past_key_values,
            encoder_outputs.hidden_states,
            encoder_outputs.attentions,
            encoder_outputs.cross_attentions);
    }

    public static async Task<RobertaModel> from_pretrained(string model_type, Device? device = null, RobertaConfig override_args = null, CancellationToken cancellation = default)
    {
        var config = await RobertaConfig.from_pretrained(model_type, cancellation);
        var roberta = new RobertaModel(config);
        if (device is not null)
        {
            roberta.to(device);
        }

        var model_tensors = roberta.state_dict();

        var ignore_buffers = new HashSet<string>
        {
            "embeddings.token_type_ids"
        };
        var model_tensors_to_load = (
            from kv in model_tensors
            where !ignore_buffers.Contains(kv.Key)
            select kv); // discard buffers, not a param
        var safeTensorsFilePath = await SafeTensors.DownloadWeightsAsync(model_type);
        var loaded_tensors = (from t in SafeTensors.LoadFile(safeTensorsFilePath, device)
            select new KeyValuePair<string, Tensor>(t.Name, t.Tensor)).ToDictionary(k => k.Key, k => k.Value);

        foreach (var (name, target_tensor) in model_tensors_to_load)
        {
            var loaded_tensor = loaded_tensors.GetValueOrDefault(name);
            Contract.Assert(loaded_tensor is not null, $"{name} tensor not found");
            // vanilla copy over the other parameters
            Contract.Assert(loaded_tensor.shape.SequenceEqual(target_tensor.shape), $"Size of loaded tensor {name}: ({loaded_tensor.shape.Stringify()}) does not match configured ({target_tensor.shape.Stringify()})");
            using (var _ = torch.no_grad())
            {
                target_tensor.copy_(loaded_tensor);
            }
        }

        roberta.eval();
        return roberta;
    }

    private Tensor invert_attention_mask(Tensor encoder_attention_mask, ScalarType? dtype = null)
    {
        Tensor encoder_extended_attention_mask;
        if (encoder_attention_mask.dim() == 3)
        {
            encoder_extended_attention_mask = encoder_attention_mask[.., TensorIndex.Null, .., ..];
        }
        else if (encoder_attention_mask.dim() == 2)
        {
            encoder_extended_attention_mask = encoder_attention_mask[.., TensorIndex.Null, TensorIndex.Null, ..];
        }
        else
        {
            throw new ArgumentException($"Bad encoder attention mask shape {encoder_attention_mask.shape}", nameof(encoder_attention_mask));
        }

        // T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        // Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        // /transformer/transformer_layers.py#L270
        // encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        // encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype ?? encoder_attention_mask.dtype); // fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_extended_attention_mask.dtype).min;
        return encoder_extended_attention_mask;
    }

    private Tensor? get_head_mask(Tensor? head_mask, int num_hidden_layers)
    {
        if (head_mask is not null)
        {
            head_mask = this._convert_head_mask_to_5d(head_mask, num_hidden_layers);
        }
        else
        {
            return null;
        }

        return head_mask;
    }

    /// <summary>
    /// -> [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    /// </summary>
    /// <param name="head_mask"></param>
    /// <param name="num_hidden_layers"></param>
    /// <param name="dtype">Force a data type (optional).</param>
    /// <returns></returns>
    private Tensor _convert_head_mask_to_5d(Tensor head_mask, int num_hidden_layers, ScalarType? dtype = null)
    {
        if (head_mask.dim() == 1)
        {
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1);
        }
        else if (head_mask.dim() == 2)
        {
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1); // We can specify head_mask for each layer
        }

        Contract.Assert(head_mask.dim() == 5, $"head_mask.dim != 5, instead {head_mask.dim()}");
        if (dtype != null)
        {
            head_mask = head_mask.to(dtype.Value); // switch to float if need + fp16 compatibility
        }

        return head_mask;
    }

    private void post_init()
    {
        if (this.config.gradient_checkpointing)
        {
            throw new NotSupportedException("post_init only needed for gradient_checkpointing which is not supported.");
        }
    }

    /// <summary>
    /// Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    /// </summary>
    /// <param name="attention_mask">Mask with ones indicating tokens to attend to, zeros for tokens to ignore.</param>
    /// <param name="input_shape">The shape of the input to the model.</param>
    /// <param name="device"></param>
    /// <param name="dtype"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    /// <exception cref="ArgumentException"></exception>
    private Tensor get_extended_attention_mask(Tensor attention_mask, Tensor input_shape, Device? device = null, ScalarType? dtype = ScalarType.Float32)
    {
        var data_type = dtype ?? attention_mask.dtype;
        var device_target = device ?? attention_mask.device;

        Tensor extended_attention_mask;
        if (attention_mask.dim() == 3)
        {
            extended_attention_mask = attention_mask[.., TensorIndex.Null, .., ..];
        }
        else if (attention_mask.dim() == 2)
        {
            if (this.config.is_decoder is true)
            {
                // TODO: extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(input_shape, attention_mask, device);
                throw new NotSupportedException("Not supported yet");
            }
            else
            {
                extended_attention_mask = attention_mask[.., TensorIndex.Null, TensorIndex.Null, ..];
            }
        }
        else
        {
            throw new ArgumentException($"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})", nameof(attention_mask));
        }

        // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        // masked positions, this operation will create a tensor which is 0.0 for
        // positions we want to attend and the dtype's smallest value for masked positions.
        // Since we are adding it to the raw scores before the softmax, this is
        // effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(data_type); // fp16 compatibility;
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(data_type).min;
        return extended_attention_mask;

/*
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """


*/
    }

    public Tensor sentence_embeddings(Tensor input_ids, Tensor attention_mask)
    {
        using (torch.no_grad())
        {
            var model_output = this.call(new RobertaModelArgs(input_ids, attention_mask));
            var sentence_embeddings = model_output.last_hidden_state.mean_pooling(attention_mask);
            return sentence_embeddings.normalize(p: 2, dim: 1);
        }
    }
}