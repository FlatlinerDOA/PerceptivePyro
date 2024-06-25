namespace PerceptivePyro.GPT;

/// <summary>
/// Multiple heads of self attention, running in parallel.
/// </summary>
internal class FeedForward : nn.Module<Tensor, Tensor>
{
    private Sequential net;

    internal FeedForward(int n_embd, double dropout) : base(nameof(FeedForward))
    {
        net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout));
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) => net.call(input);
}
