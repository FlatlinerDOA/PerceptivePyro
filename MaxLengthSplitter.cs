namespace PerceptivePyro;

public sealed class MaxLengthSplitter : ITokenSplitter
{
    public MaxLengthSplitter(int maxLength)
    {
        this.MaxLength = maxLength;
    }

    public int MaxLength { get; init; }

    public IEnumerable<IReadOnlyList<int>> Split(IEnumerable<int> tokens) => tokens.Paginate(this.MaxLength);
}