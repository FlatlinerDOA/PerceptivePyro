namespace NanoGPTSharp;

/// <summary>
/// Transformer block: Communication between tokens followed by computation on tokens.
/// </summary>
internal class Block : nn.Module<Tensor, Tensor>
{
    public int head_size;
    public CausalSelfAttention attn;
    public MultiLayerPerceptron mlp;
    public LayerNorm ln1;
    public LayerNorm ln2;

    public Block(GPTConfig config): this(config.n_embd, config.n_head, config.block_size, config.dropout, config.has_bias)
    {
    }

    internal Block(int n_embd, int n_heads, int block_size, double dropout, bool hasBias) : base(nameof(Block))
    {
        this.head_size = n_embd / n_heads;
        this.ln1 = new LayerNorm(n_embd, hasBias);
        this.attn = new CausalSelfAttention(n_embd, n_heads, block_size, dropout, hasBias);
        this.mlp = new MultiLayerPerceptron(n_embd, dropout, hasBias);
        this.ln2 = new LayerNorm(n_embd, hasBias);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Pre-norm
        x = x + this.attn.call(this.ln1.call(x));
        x = x + this.mlp.call(this.ln2.call(x));
        return x;
    }
}
