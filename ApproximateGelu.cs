namespace PerceptivePyro;

public class ApproximateGelu : nn.Module<Tensor, Tensor>
{
    private static readonly double sqrt_2_pi = Math.Sqrt(2.0 / Math.PI);

    public ApproximateGelu(string name = "gelu") : base(name)
    {
    }

    /// <summary>
    /// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    /// Reference: <a href="https://arxiv.org/abs/1606.08415">Gaussian Error Linear Units(GELU) paper</a>
    /// </summary>
    /// <param name="x">Input tensor</param>
    /// <returns>Tensor of gelu operator applied.</returns>
    private static Tensor Gelu(Tensor x) => 0.5 * x * (1.0 + torch.tanh(sqrt_2_pi * (x + 0.044715 * torch.pow(x, 3.0))));

    /// <inheritdoc/>
    public override Tensor forward(Tensor x)
    {
        return Gelu(x);
    }
}