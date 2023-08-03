using System.Diagnostics;

namespace PerceptivePyro;

using F = nn.functional;

internal class CausalSelfAttention : nn.Module<Tensor, Tensor>
{
    public Linear c_attn;
    public Linear c_proj;
    public Dropout attn_dropout;
    public Dropout resid_dropout;
    public int n_head;
    public int n_embd;
    public double dropout;
    public bool hasFlash;

    public CausalSelfAttention(GPTConfig config) : this(config.n_embd, config.n_head, config.block_size, config.dropout, config.has_bias)
    {
    }

    public CausalSelfAttention(int n_embd, int n_head, int block_size, double dropout, bool hasBias) : base(nameof(CausalSelfAttention))
    {
        Debug.Assert(n_embd % n_head == 0, "n_embd must be an even multiple of n_head");

        this.c_attn = nn.Linear(n_embd, 3 * n_embd, hasBias: hasBias);
        this.c_proj = nn.Linear(n_embd, n_embd, hasBias: hasBias);
        this.attn_dropout = nn.Dropout(dropout);
        this.resid_dropout = nn.Dropout(dropout);
        this.n_head = n_head;
        this.n_embd = n_embd;
        this.dropout = dropout;
        // flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        this.hasFlash = typeof(nn.functional).GetMember("scaled_dot_product_attention").Length != 0;
        if (!this.hasFlash)
        {
            Debug.WriteLine("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0");

            // causal mask to ensure that attention is only applied to the left in the input sequence
            this.bias = torch.tril(torch.ones(block_size, block_size))
                .view(1, 1, block_size, block_size);
        }

        this.RegisterComponents();
    }

    /// <summary>
    /// Gets or sets the bias buffer.
    /// </summary>
    public Tensor bias
    {
        get => this.get_buffer("bias");
        set => this.register_buffer("bias", value);
    }

    public override Tensor forward(Tensor input)
    {
        var (B, T, C) = (input.shape[0], input.shape[1], input.shape[2]);
        var p = this.c_attn.call(input).split(this.n_embd, dim: 2);
        var (q, k, v) = (p[0], p[1], p[2]);
        k = k.view(B, T, this.n_head, C / this.n_head).transpose(1, 2); // (B, nh, T, hs)
        q = q.view(B, T, this.n_head, C / this.n_head).transpose(1, 2); // (B, nh, T, hs)
        v = v.view(B, T, this.n_head, C / this.n_head).transpose(1, 2); // (B, nh, T, hs)

        Tensor y;
        if (this.hasFlash)
        {
            // TODO: Add this when TorchSharp upgrades to 2.0
            throw new Exception("Adopt flash attention support now!");
        }
        else
        {
            // manual implementation of attention
            var att = q.matmul(k.transpose(-2, -1)) * (1.0d / Math.Sqrt((double)k.size(-1))); // (B, nh, T, hs) -> (B, nh, hs, T)
            att = att.masked_fill(this.bias[.., .., ..(int)T, ..(int)T] == 0, float.NegativeInfinity);
            att = F.softmax(att, dim: -1);
            att = this.attn_dropout.call(att);
            y = att.matmul(v); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        }

        y = y.transpose(1, 2).contiguous().view(B, T, C); // re-assemble all head outputs side by side

        // output projection
        y = this.resid_dropout.call(this.c_proj.call(y));
        return y;
    }
}