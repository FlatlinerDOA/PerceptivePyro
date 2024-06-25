namespace PerceptivePyro;

/// <summary>
/// Multi Layer Perceptron or (MLP).
/// Essentially a linear transformation, activation function and linear transformation. With optional dropout.
/// </summary>
internal class MultiLayerPerceptron : nn.Module<Tensor, Tensor>
{
    private Linear c_fc;
    private GELU gelu;
    private Linear c_proj;
    private Dropout? dropout;

    public MultiLayerPerceptron(int n_embd, double dropout, bool hasBias) : base(nameof(MultiLayerPerceptron))
    {
        this.c_fc = nn.Linear(n_embd, 4 * n_embd, hasBias: hasBias);
        // NOTE: Can swap out with ApproximateGelu to the match GPT paper for training,
        // but GELU seems to give identical results for inference.
        this.gelu = nn.GELU(); // new ApproximateGelu();
        this.c_proj = nn.Linear(4 * n_embd, n_embd, hasBias: hasBias);
        this.dropout = dropout > 0.0d ? nn.Dropout(dropout) : null;
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = this.c_fc.call(x);
        x = this.gelu.call(x);
        x = this.c_proj.call(x);
        x = this.dropout != null ? this.dropout.call(x) : x;
        return x;
    }
}