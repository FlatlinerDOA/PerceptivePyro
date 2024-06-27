using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;
namespace PerceptivePyro.Whisper;

/// <summary>
/// Extends GPT with language tokens.
/// </summary>
public class WhisperEncoding
{
    private readonly Tokenizer encoding;
    public WhisperEncoding(string name, int explicit_n_vocab, Regex pat_str, IReadOnlyDictionary<byte[], int> mergeable_ranks, IReadOnlyDictionary<string, int> special_tokens)
    {
        // TODO: Load BPE encoding.
        this.encoding = Tokenizer.CreateTiktokenForModel(name, special_tokens);
        this.eot_token = this.encoding.EncodeToIds("<|endoftext|>")[0];
    }

    public static WhisperEncoding GetEncoding(string name = "gpt2", int num_languages = 99)
    {
        var vocab_path = Path.Combine(Environment.CurrentDirectory, "assets", $"{name}.tiktoken");
        var ranks = new Dictionary<byte[], int>(
            from line in File.ReadLines(vocab_path)
            where !string.IsNullOrEmpty(line)
            let tokenrank = line.Split(' ')
            select new KeyValuePair<byte[], int>(Convert.FromBase64String(tokenrank[0]), int.Parse(tokenrank[1])));

        var special_tokens = new Dictionary<string, int>();
        var n_vocab = ranks.Count;
        string[] specials =
        [
            "<|endoftext|>",
            "<|startoftranscript|>",
            .. from lang in WhisperTokenizer.Languages.Keys.Take(num_languages) select $"<|{lang}|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            .. from i in Enumerable.Range(0, 1501) select $"<|{i * 0.02:.2f}|>",
        ];

        foreach (var token in specials)
        {
            special_tokens[token] = n_vocab;
            n_vocab += 1;
        }

        return new WhisperEncoding(
            Path.GetFileNameWithoutExtension(vocab_path),
            n_vocab,
            new Regex(@"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""),
            ranks,
            special_tokens);
    }

    public IReadOnlyList<int> Encode(string text) => this.encoding.EncodeToIds(text);

    public string? Decode(IReadOnlyList<int> text) => this.encoding.Decode(text);

    public int eot_token { get; init; }
}