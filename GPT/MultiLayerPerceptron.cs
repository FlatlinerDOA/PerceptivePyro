namespace PerceptivePyro.GPT;

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
        c_fc = nn.Linear(n_embd, 4 * n_embd, hasBias: hasBias);
        // NOTE: Can swap out with ApproximateGelu to the match GPT paper for training,
        // but GELU seems to give identical results for inference.
        gelu = nn.GELU(); // new ApproximateGelu();
        c_proj = nn.Linear(4 * n_embd, n_embd, hasBias: hasBias);
        this.dropout = dropout > 0.0d ? nn.Dropout(dropout) : null;
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = c_fc.call(x);
        x = gelu.call(x);
        x = c_proj.call(x);
        x = dropout != null ? dropout.call(x) : x;
        return x;
    }
}