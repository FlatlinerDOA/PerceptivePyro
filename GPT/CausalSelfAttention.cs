namespace PerceptivePyro.GPT;

using System.Diagnostics;
using TorchSharp.Modules;
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

        c_attn = nn.Linear(n_embd, 3 * n_embd, hasBias: hasBias);
        c_proj = nn.Linear(n_embd, n_embd, hasBias: hasBias);
        attn_dropout = nn.Dropout(dropout);
        resid_dropout = nn.Dropout(dropout);
        this.n_head = n_head;
        this.n_embd = n_embd;
        this.dropout = dropout;
        // flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        hasFlash = typeof(F).GetMember("scaled_dot_product_attention").Length != 0;
        if (!hasFlash)
        {
            Debug.WriteLine("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0");

            // causal mask to ensure that attention is only applied to the left in the input sequence
            register_buffer("bias", tril(ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size));
        }

        RegisterComponents();
    }

    /// <summary>
    /// Gets or sets the bias buffer.
    /// </summary>
    public Tensor bias
    {
        get => get_buffer("bias");
        set => register_buffer("bias", value);
    }

    public override Tensor forward(Tensor input)
    {
        var (B, T, C) = (input.shape[0], input.shape[1], input.shape[2]);
        var qkv = c_attn.call(input);
        var p = qkv.split(n_embd, dim: 2);
        var (q, k, v) = (p[0], p[1], p[2]);
        k = k.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)
        q = q.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)
        v = v.view(B, T, n_head, C / n_head).transpose(1, 2); // (B, nh, T, hs)

        Tensor y;
        if (hasFlash)
        {
            y = F.scaled_dot_product_attention(q, k, v, attn_mask: null, p: training ? dropout : 0, is_casual: true);
        }
        else
        {
            // manual implementation of attention
            var att = q.matmul(k.transpose(-2, -1)) * (1.0d / Math.Sqrt(k.size(-1))); // (B, nh, T, hs) -> (B, nh, hs, T)
            att = att.masked_fill(bias[.., .., ..(int)T, ..(int)T] == 0, float.NegativeInfinity);
            att = F.softmax(att, dim: -1);
            att = attn_dropout.call(att);
            y = att.matmul(v); // (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        }

        y = y.transpose(1, 2).contiguous().view(B, T, C); // re-assemble all head outputs side by side

        // output projection
        y = resid_dropout.call(c_proj.call(y));
        return y;
    }
}

