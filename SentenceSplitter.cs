namespace PerceptivePyro;

public sealed class SentenceSplitter : ITokenSplitter
{
    private static readonly int[] DefaultSplitTokens = new[]
    {
        50118, // \n
        4      // .
    };

    public SentenceSplitter(int maxLength, int[] splitTokens = null)
    {
        this.MaxLength = maxLength;
        this.SplitTokens = splitTokens ?? DefaultSplitTokens;
    }

    public int[] SplitTokens { get; init; }

    public int MaxLength { get; init; }

    public IEnumerable<IReadOnlyList<int>> Split(IEnumerable<int> tokens)
    {
        var remainder = tokens.ToArray().AsSpan();
        
        // Remove BOS and EOS, we will re-insert them for each stream.
        var bos = remainder[0..1];
        var eos = remainder[^1..];
        remainder = remainder[1..^1];
        
        var list = new List<IReadOnlyList<int>>();
        var s = this.SplitTokens.AsSpan();
        while (remainder.Length > 0)
        {
            var max = Math.Min(this.MaxLength, remainder.IndexOfAny(s));
            if (max <= 0)
            {
                max = Math.Min(this.MaxLength, remainder.Length);
            }

            list.Add(bos.Concat(remainder.Slice(0, max), eos));
            remainder = remainder.Slice(max);
        }

        return list;
    }
}