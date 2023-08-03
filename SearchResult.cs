namespace PerceptivePyro;

public sealed record SearchResult<TKey, TValue>(TKey Key, TValue Value, string Excerpt, float Score);