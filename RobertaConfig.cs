namespace NanoGPTSharp;

public record RobertaConfig
{
    public bool output_hidden_states { get; init; } = false;
    public bool output_attentions { get; init; } = false;
    public string _name_or_path { get; init; } = "sentence-transformers/all-distilroberta-v1";
    public IReadOnlyList<string> architectures { get; init; } = new string[] { "RobertaForMaskedLM" };
    public double attention_probs_dropout_prob { get; init; } = 0.1;
    public int bos_token_id { get; init; } = 0;
    public object classifier_dropout { get; init; } = null;
    public int eos_token_id { get; init; } = 2;
    public bool gradient_checkpointing { get; init; } = false;
    
    /// <summary>
    /// Gets or inits an activation function or a standard named function e.g "gelu" or "relu" etc.
    /// </summary>
    public object hidden_act { get; init; } = "gelu";
    public double hidden_dropout_prob { get; init; } = 0.1;
    public int hidden_size { get; init; } = 768;
    public double initializer_range { get; init; } = 0.02;
    public int intermediate_size { get; init; } = 3072;
    public double layer_norm_eps { get; init; } = 1e-05;
    public int max_position_embeddings { get; init; } = 514;
    public string model_type { get; init; } = "roberta";
    public int num_attention_heads { get; init; } = 12;
    public int num_hidden_layers { get; init; } = 6;
    public int pad_token_id { get; init; } = 1;
    public string position_embedding_type { get; init; } = "absolute";
    public string transformers_version { get; init; } = "4.28.1";
    public int type_vocab_size { get; init; } = 1;
    public bool use_cache { get; init; } = true;
    public int vocab_size { get; init; } = 50265;

    public int? embedding_size { get; init; }
    public bool? is_decoder { get; init; }
    public int chunk_size_feed_forward { get; }
    public bool? add_cross_attention { get; }
}
