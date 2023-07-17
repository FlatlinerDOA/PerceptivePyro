namespace PerceptivePyro;

/// <summary>
/// Transformer block: Communication between tokens followed by computation on tokens.
/// </summary>
internal class TransformerBlock : nn.Module<Tensor, Tensor>
{
    public LayerNorm ln_1;
    public CausalSelfAttention attn;
    public LayerNorm ln_2;
    public MultiLayerPerceptron mlp;

    public TransformerBlock(GPTConfig config): this(config.n_embd, config.n_head, config.block_size, config.dropout, config.has_bias)
    {
    }

    internal TransformerBlock(int n_embd, int n_heads, int block_size, double dropout, bool hasBias) : base(nameof(TransformerBlock))
    {
        this.ln_1 = new LayerNorm(n_embd, hasBias);
        this.attn = new CausalSelfAttention(n_embd, n_heads, block_size, dropout, hasBias);
        this.ln_2 = new LayerNorm(n_embd, hasBias);
        this.mlp = new MultiLayerPerceptron(n_embd, dropout, hasBias);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Pre-norm
        x = x + this.attn.call(this.ln_1.call(x));
        x = x + this.mlp.call(this.ln_2.call(x));
        return x;
    }
}
