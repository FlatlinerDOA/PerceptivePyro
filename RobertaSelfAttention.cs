namespace NanoGPTSharp;

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;

public record struct RobertaSelfAttentionArgs(Tensor? attention_mask = null, Tensor? head_mask = null, Tensor? encoder_hidden_states = null, Tensor? encoder_attention_mask = null, IReadOnlyList<Tensor>? past_key_value = null, bool? output_attentions = null);
////public record RobertaSelfAttentionOutput(Tensor context_layer, Tensor? attention_probs, IReadOnlyList<Tensor>? past_key_value);
public partial class RobertaSelfAttention : nn.Module<Tensor, RobertaSelfAttentionArgs, IReadOnlyList<Tensor>>
{
    public readonly int num_attention_heads;
    public readonly int attention_head_size;
    public readonly int all_head_size;
    public readonly Linear query;
    public readonly Linear key;
    public readonly Linear value;
    public readonly Dropout dropout;
    public readonly string position_embedding_type;
    public readonly int max_position_embeddings;
    public readonly Embedding distance_embedding;
    public readonly bool is_decoder;

    public RobertaSelfAttention(RobertaConfig config, string? position_embedding_type = null) : base(nameof(RobertaSelfAttention))
    {
        Contract.Assert(
            config.hidden_size % config.num_attention_heads == 0 || config.embedding_size is not null,
            $"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})");

        this.num_attention_heads = config.num_attention_heads;
        this.attention_head_size = (int)(config.hidden_size / config.num_attention_heads);
        this.all_head_size = this.num_attention_heads * this.attention_head_size;

        this.query = nn.Linear(config.hidden_size, this.all_head_size);
        this.key = nn.Linear(config.hidden_size, this.all_head_size);
        this.value = nn.Linear(config.hidden_size, this.all_head_size);

        this.dropout = nn.Dropout(config.attention_probs_dropout_prob);
        this.position_embedding_type = position_embedding_type ?? config.position_embedding_type ?? "absolute";
        if (this.position_embedding_type == "relative_key" || this.position_embedding_type == "relative_key_query") {
            this.max_position_embeddings = config.max_position_embeddings;
            this.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, this.attention_head_size);
        }

        this.is_decoder = config.is_decoder ?? false;
        this.RegisterComponents();
    }

    private Tensor transpose_for_scores(Tensor x)
    {
        var new_x_shape = x.size()[..-1].Concat(new long[] { this.num_attention_heads, this.attention_head_size }).ToArray();
        x = x.view(new_x_shape);
        return x.permute(0, 2, 1, 3);
    }

    public override IReadOnlyList<Tensor> forward(Tensor hidden_states, RobertaSelfAttentionArgs optional)
    {
        var (attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions) = optional;
        var mixed_query_layer = this.query.call(hidden_states);

        // If this is instantiated as a cross-attention module, the keys
        // and values come from an encoder; the attention mask needs to be
        // such that the encoder's padding tokens are not attended to.
        Tensor key_layer;
        Tensor value_layer;
        var is_cross_attention = encoder_hidden_states is not null;
        if (is_cross_attention && past_key_value is not null)
        {
            // reuse k,v, cross_attentions
            key_layer = past_key_value[0];
            value_layer = past_key_value[1];
            attention_mask = encoder_attention_mask;
        }
        else if (is_cross_attention)
        {
            key_layer = this.transpose_for_scores(this.key.call(encoder_hidden_states));
            value_layer = this.transpose_for_scores(this.value.call(encoder_hidden_states));
            attention_mask = encoder_attention_mask;
        }
        else if (past_key_value is not null)
        {
            key_layer = this.transpose_for_scores(this.key.call(hidden_states));
            value_layer = this.transpose_for_scores(this.value.call(hidden_states));
            key_layer = torch.cat(new[] { past_key_value[0], key_layer }, dim: 2);
            value_layer = torch.cat(new[] { past_key_value[1], value_layer }, dim: 2);
        }
        else
        {
            key_layer = this.transpose_for_scores(this.key.call(hidden_states));
            value_layer = this.transpose_for_scores(this.value.call(hidden_states));
        }

        var query_layer = this.transpose_for_scores(mixed_query_layer);

        var use_cache = past_key_value is not null;

        if (this.is_decoder)
        {
            // if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            // Further calls to cross_attention layer can then reuse all cross-attention
            // key/value_states (first "if" case)
            // if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            // all previous decoder key/value_states. Further calls to uni-directional self-attention
            // can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            // if encoder bi-directional self-attention `past_key_value` is always `null`
            past_key_value = new[] { key_layer, value_layer };
        }

        // Take the dot product between "query" and "key" to get the raw attention scores.
        var attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2));
        Tensor position_ids_l;
        if (this.position_embedding_type == "relative_key" || this.position_embedding_type == "relative_key_query")
        {
            var (query_length, key_length) = (query_layer.shape[2], key_layer.shape[2]);
            if (use_cache)
            {
                position_ids_l = torch.tensor(key_length - 1, dtype: torch.@long, device: hidden_states.device).view(-1, 1);
            }
            else
            {
                position_ids_l = torch.arange(query_length, dtype: torch.@long, device: hidden_states.device).view(-1, 1);
            }

            var position_ids_r = torch.arange(key_length, dtype: torch.@long, device: hidden_states.device).view(1, -1);
            var distance = position_ids_l - position_ids_r;

            var positional_embedding = this.distance_embedding.call(distance + this.max_position_embeddings - 1);
            positional_embedding = positional_embedding.to(query_layer.dtype); // fp16 compatibility

            Tensor relative_position_scores;
            Tensor relative_position_scores_query;
            Tensor relative_position_scores_key;
            if (this.position_embedding_type == "relative_key")
            {
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding);
                attention_scores = attention_scores + relative_position_scores;
            }
            else if (this.position_embedding_type == "relative_key_query")
            {
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding);
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding);
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key;
            }
        }

        attention_scores = attention_scores / Math.Sqrt(this.attention_head_size);
        if (attention_mask is not null)
        {
            // Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask;
        }

        // Normalize the attention scores to probabilities.
        var attention_probs = nn.functional.softmax(attention_scores, dim: -1);

        // This is actually dropping out entire tokens to attend to, which might
        // seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = this.dropout.call(attention_probs);

        // Mask heads if we want to
        if (head_mask is not null)
        {
            attention_probs = attention_probs * head_mask;
        }

        var context_layer = torch.matmul(attention_probs, value_layer);
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous();
        var new_context_layer_shape = context_layer.size()[..-2].Append(this.all_head_size).ToArray();
        context_layer = context_layer.view(new_context_layer_shape);
        var outputs = new List<Tensor>(3)
        {
            context_layer
        };
        if (output_attentions is true) outputs.Add(attention_probs);
        if (this.is_decoder && past_key_value is not null) outputs.AddRange(past_key_value);
        return outputs;
    }
}