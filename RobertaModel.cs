namespace NanoGPTSharp;

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp.Modules;
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

public class RobertaModel : nn.Module<RobertaModelArgs, (Tensor logits, Tensor? loss)>
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

        // Initialize weights and apply final processing
        this.post_init();

        this.RegisterComponents();
    }

    public override (Tensor logits, Tensor? loss) forward(RobertaModelArgs input)
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

        Contract.Assert(input_ids is null || inputs_embeds is null, "You cannot specify both input_ids and inputs_embeds at the same time");

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
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device: device);
        }

        if (token_type_ids is null)
        {
            if (this.embeddings.has_buffer("token_type_ids"))
            {
                buffered_token_type_ids = this.embeddings.token_type_ids[.., ..(int)seq_length];
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length);
                token_type_ids = buffered_token_type_ids_expanded;
            }
            else
            {
                token_type_ids = torch.zeros(input_shape, dtype: torch.@long, device=device);
            }
        }

        // We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        // ourselves in which case we just need to make it broadcastable to all heads.
        var extended_attention_mask = this.get_extended_attention_mask(attention_mask, input_shape);

        // If a 2D or 3D attention mask is provided for the cross-attention
        // we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if (this.config.is_decoder is true && encoder_hidden_states is not null)
        {
            var (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size();
            var encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length);
            if (encoder_attention_mask is null)
            {
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device: device);
            }

            var encoder_extended_attention_mask = this.invert_attention_mask(encoder_attention_mask);
        }
        else
        {
            var encoder_extended_attention_mask = null;
        }
    /*
        
        // Prepare head mask if needed
        // 1.0 in head_mask indicate we keep the head
        // attention_probs has shape bsz x n_heads x N x N
        // input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        // and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = this.get_head_mask(head_mask, this.config.num_hidden_layers);

        embedding_output = this.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = this.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = this.pooler(sequence_output) if this.pooler is not null else null
 */
    }
}
