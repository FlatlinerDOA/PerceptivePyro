namespace PerceptivePyro;

/// <summary>
/// MLP
/// </summary>
internal class MultiLayerPerceptron : nn.Module<Tensor, Tensor>
{
    private static readonly double sqrt_2_pi = Math.Sqrt(2.0 / Math.PI);
    private Linear c_fc;
    private Linear c_proj;
    private Dropout dropout;

    public MultiLayerPerceptron(int n_embd, double dropout, bool hasBias) : base(nameof(MultiLayerPerceptron))
    {
        this.c_fc = nn.Linear(n_embd, 4 * n_embd, hasBias: hasBias);
        this.c_proj = nn.Linear(4 * n_embd, n_embd, hasBias: hasBias);
        this.dropout = nn.Dropout(dropout);
        this.RegisterComponents();
    }

    /// <summary>
    /// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    /// Reference: <a href="https://arxiv.org/abs/1606.08415">Gaussian Error Linear Units(GELU) paper</a>
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <returns>Tensor of gelu operator applied.</returns>
    private static Tensor NewGelu(Tensor x) => 0.5 * x * (1.0 + torch.tanh(sqrt_2_pi * (x + 0.044715 * torch.pow(x, 3.0))));

    public override Tensor forward(Tensor x)
    {
        x = this.c_fc.call(x);
        x = NewGelu(x);
        x = this.c_proj.call(x);
        x = this.dropout.call(x);
        return x;
    }
}
