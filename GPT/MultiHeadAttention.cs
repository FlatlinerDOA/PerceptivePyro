namespace PerceptivePyro.GPT;

/// <summary>
/// Multiple heads of self attention, running in parallel.
/// </summary>
internal class MultiHeadAttention : nn.Module<Tensor, Tensor>
{
    private ModuleList<Head> heads;
    private Linear proj;
    private Dropout dropout;

    internal MultiHeadAttention(int num_heads, int block_size, int n_embd, int head_size, double dropout) : base(nameof(Head))
    {
        heads = nn.ModuleList(Enumerable.Range(0, num_heads).Select(h => new Head(block_size, n_embd, head_size, dropout)).ToArray());
        proj = nn.Linear(n_embd, n_embd);
        this.dropout = nn.Dropout(dropout);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var output = cat(heads.Select(h => h.call(input)).ToList(), dim: -1);
        output = proj.call(output);
        output = dropout.call(output);
        return output;
    }

}
