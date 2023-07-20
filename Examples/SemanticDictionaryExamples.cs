using System.Diagnostics;

namespace PerceptivePyro.Examples;

internal class SemanticDictionaryExamples
{
    /// <summary>
    /// Indexes a bunch of sentences into a dictionary and then looks up from a similar sentence.
    /// </summary>
    /// <returns></returns>
    public static async Task Semantic_Dictionary_TopK()
    {
        var dict = new SemanticDictionary<string, string>(k => k);
        await dict.InitializeAsync();

        dict.AddAll(new KeyValuePair<string, string>[]
        {
            new("1", "The quick brown fox"),
            new("2", "Jumps over the lazy dog"),
            new("3", "All the king's horses")
        });

        var s = Stopwatch.StartNew();
        var results = dict.GetBatchTop(new[] { "lazy dogs", "queens chariots" });
        s.ElapsedMilliseconds.Dump();
        results.Dump();
        
        dict.Remove("2");
        
        var results2 = dict.GetBatchTop(new[] { "lazy dogs", "queens chariots" });
        
        results2.Dump();
    }
}