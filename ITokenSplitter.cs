namespace PerceptivePyro;

/// <summary>
/// Splitter should always include the splitting token at the end.
/// </summary>
public interface ITokenSplitter
{
    IEnumerable<IReadOnlyList<int>> Split(IEnumerable<int> tokens);
}