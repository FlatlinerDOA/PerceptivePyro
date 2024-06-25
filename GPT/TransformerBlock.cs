namespace PerceptivePyro.GPT;

/// <summary>
/// Transformer block: Communication between tokens followed by computation on tokens.
/// </summary>
internal class TransformerBlock : nn.Module<Tensor, Tensor>
{
    public LayerNorm ln_1;
    public CausalSelfAttention attn;
    public LayerNorm ln_2;
    public MultiLayerPerceptron mlp;

    public TransformerBlock(GPTConfig config) : this(config.n_embd, config.n_head, config.block_size, config.dropout, config.has_bias)
    {
    }

    internal TransformerBlock(int n_embd, int n_heads, int block_size, double dropout, bool hasBias) : base(nameof(TransformerBlock))
    {
        ln_1 = new LayerNorm(n_embd, hasBias);
        attn = new CausalSelfAttention(n_embd, n_heads, block_size, dropout, hasBias);
        ln_2 = new LayerNorm(n_embd, hasBias);
        mlp = new MultiLayerPerceptron(n_embd, dropout, hasBias);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Pre-norm
        x = x + attn.call(ln_1.call(x));
        x = x + mlp.call(ln_2.call(x));
        return x;
    }
}
