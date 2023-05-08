namespace NanoGPTSharp;
using SharpToken;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Text;
using static TorchSharp.torch;
using System.Diagnostics;
using System.Text.Json;
using TorchSharp.Modules;

public class RobertaTokenizer
{
    private const string RobertaPattern = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"; //r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
    private readonly BytePairEncodingCore bytePairEncoding;
    private readonly Dictionary<string, int> specialTokenMappings;

    public int MaxTokenValue { get; init; }

    private RobertaTokenizer(string patternString, Dictionary<byte[], int> bytePairRanks, Dictionary<string, int> specialTokenMappings, int? explicitNVocab = null)
    {
        this.MaxTokenValue = Math.Max(GetMaxValueFromDictionary(bytePairRanks), GetMaxValueFromDictionary(specialTokenMappings));
        this.specialTokenMappings = specialTokenMappings;
        if (explicitNVocab.HasValue)
        {
            if (bytePairRanks.Count + specialTokenMappings.Count != explicitNVocab.Value)
            {
                throw new ArgumentException("The number of mergeable tokens and special tokens must be equal to explicit_n_vocab.");
            }

            if (this.MaxTokenValue != explicitNVocab.Value - 1)
            {
                throw new ArgumentException("The maximum token value must be equal to explicit_n_vocab - 1.");
            }
        }

        this.bytePairEncoding = new BytePairEncodingCore(bytePairRanks, specialTokenMappings, new Regex(patternString, RegexOptions.Compiled));
    }

    private static string SpecialTokenRegex(ISet<string> tokens)
    {
        List<string> list = new List<string>();
        foreach (string token in tokens)
        {
            list.Add(Regex.Escape(token));
        }

        string text = string.Join("|", list);
        return "(" + text + ")";
    }

    public List<int> Encode(string lineToEncode, ISet<string>? allowedSpecial = null, ISet<string>? disallowedSpecial = null)
    {
        HashSet<string> hashSet = new HashSet<string>(this.specialTokenMappings.Keys);
        allowedSpecial ??= new HashSet<string>();
        disallowedSpecial ??= new HashSet<string> { "all" };
        if (disallowedSpecial.Contains("all"))
        {
            disallowedSpecial = new HashSet<string>(hashSet);
            disallowedSpecial.ExceptWith(allowedSpecial);
        }

        if (allowedSpecial.Contains("all"))
        {
            allowedSpecial = hashSet;
        }

        if (disallowedSpecial.Count > 0)
        {
            string pattern = SpecialTokenRegex(new HashSet<string>(disallowedSpecial));
            Match match = Regex.Match(lineToEncode, pattern);
            if (match.Success)
            {
                throw new ArgumentException("Disallowed special token found: " + match.Value);
            }
        }

        return new int[] { this.specialTokenMappings.GetValueOrDefault("<s>") }.Concat(this.bytePairEncoding.EncodeNative(lineToEncode, allowedSpecial).Item1).Concat(new int[] { this.specialTokenMappings.GetValueOrDefault("</s>") }).ToList();
    }

    public string Decode(List<int> inputTokensToDecode, bool trimSentenceTokens = true)
    {
        List<byte> list = this.bytePairEncoding.DecodeNative(inputTokensToDecode.ToArray());
        var decoded = Encoding.UTF8.GetString(list.ToArray());
        if (trimSentenceTokens)
        {
            return decoded.Replace("<s>", string.Empty).Replace("</s>", string.Empty);
        }
        
        return decoded;
    }

    private static int GetMaxValueFromDictionary(Dictionary<byte[], int> dictionary)
    {
        return dictionary.Values.Prepend(0).Max();
    }

    private static int GetMaxValueFromDictionary(Dictionary<string, int> dictionary)
    {
        return dictionary.Values.Prepend(0).Max();
    }

    public static async Task<RobertaTokenizer> from_pretrained(string model, Device? device = null, CancellationToken cancellation = default)
    {
        var (tokens, special) = await DownloadTokensAsync(model, cancellation);
        return new RobertaTokenizer(RobertaPattern, tokens, special);
    }

    private static async Task<(Dictionary<byte[], int> BytesToTokens, Dictionary<string, int> SpecialTokens)> DownloadTokensAsync(string model, CancellationToken cancellation)
    {
        var model_path = Path.Combine(model, "tokenizer.json");
        var tokenizerFilePath = Path.GetFullPath($@".\models\{model}\tokenizer.json");
        var specialTokensFilePath = Path.GetFullPath($@".\models\{model}\special_tokens_map.json");
        if (File.Exists(tokenizerFilePath) && File.Exists(specialTokensFilePath))
        {
            return LoadFiles(tokenizerFilePath, specialTokensFilePath);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(tokenizerFilePath)!);
        Trace.TraceInformation($"Downloading weights from pretrained {model} to {tokenizerFilePath}");
        using var client = new HttpClient();
        var tokenJson = client.GetStringAsync($"https://huggingface.co/sentence-transformers/{model}/raw/main/tokenizer.json", cancellation);
        var specialJson = client.GetStringAsync($"https://huggingface.co/sentence-transformers/{model}/raw/main/special_tokens_map.json", cancellation);
        await File.WriteAllTextAsync(tokenizerFilePath, await tokenJson);
        await File.WriteAllTextAsync(specialTokensFilePath, await specialJson);
        return LoadFiles(tokenizerFilePath, specialTokensFilePath);
    }

    private static (Dictionary<byte[], int> BytesToTokens, Dictionary<string, int> SpecialTokens) LoadFiles(string tokenizerFilePath, string specialTokensFilePath)
    {
        using var tfs = File.OpenRead(tokenizerFilePath);
        var tjson = JsonDocument.Parse(tfs);

        var encoding = Encoding.UTF8;
        var tq = from prop in tjson.RootElement.EnumerateObject()
                 where prop.Name == "model"
                 from vocab in prop.Value.EnumerateObject()
                 where vocab.Name == "vocab"
                 from v in vocab.Value.EnumerateObject()
                 select new KeyValuePair<byte[], int>(encoding.GetBytes(v.Name.Replace('Ġ', ' ')), v.Value.GetInt32());
        var tqd = new Dictionary<byte[], int>(tq, new ByteArrayEqualityComparer());

        using var sfs = File.OpenRead(specialTokensFilePath);
        var sjson = JsonDocument.Parse(sfs);
        var special_tokens = (from prop in sjson.RootElement.EnumerateObject()
                              where prop.Value.ValueKind == JsonValueKind.String
                             select prop.Value.GetString() ?? string.Empty).ToHashSet();
        var sq = from special_token in special_tokens
                 select new KeyValuePair<string, int>(special_token, tqd.GetValueOrDefault(encoding.GetBytes(special_token)));
        return (tqd, new Dictionary<string, int>(sq));
    }

    internal sealed class ByteArrayEqualityComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[]? x, byte[]? y)
        {
            if (x == null || y == null)
            {
                return false;
            }

            if (x != y)
            {
                return StructuralComparisons.StructuralEqualityComparer.Equals(x, y);
            }

            return true;
        }

        public int GetHashCode(byte[] obj)
        {
            return StructuralComparisons.StructuralEqualityComparer.GetHashCode(obj);
        }
    }
}
