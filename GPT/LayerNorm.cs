namespace PerceptivePyro.GPT;

using F = nn.functional;

/// <summary>
/// LayerNorm but with an optional bias. PyTorch doesn't support simply hasBias:False
/// </summary>
internal class LayerNorm : nn.Module<Tensor, Tensor>
{
    private Parameter weight;
    private Parameter? bias;

    public LayerNorm(int ndim, bool hasBias) : base(nameof(LayerNorm))
    {
        weight = nn.Parameter(ones(ndim));
        bias = hasBias ? nn.Parameter(zeros(ndim)) : null;
    }

    public override Tensor forward(Tensor input) => F.layer_norm(input, weight.shape, weight, bias);
}
