namespace PerceptivePyro;

/// <summary>
/// MLP
/// </summary>
internal class MultiLayerPerceptron : nn.Module<Tensor, Tensor>
{
    private readonly Linear c_fc;
    private readonly Linear c_proj;
    private readonly Dropout dropout;

    public MultiLayerPerceptron(int n_embd, double dropout, bool hasBias) : base(nameof(MultiLayerPerceptron))
    {
        this.c_fc = nn.Linear(n_embd, 4 * n_embd, hasBias: hasBias);
        this.c_proj = nn.Linear(4 * n_embd, n_embd, hasBias: hasBias);
        this.dropout = nn.Dropout(dropout);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = this.c_fc.call(x);
        x = x.NewGelu();
        x = this.c_proj.call(x);
        x = this.dropout.call(x);
        return x;
    }
}
