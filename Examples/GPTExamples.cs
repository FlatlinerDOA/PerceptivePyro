using System.Diagnostics;
using System.Text.Json;
using SharpToken;

namespace PerceptivePyro.Examples;

internal class GPTExamples
{
    /// <summary>
    /// Evaluates GPT2 embedding sentence similarity scoring on the MS MARCO V2.1 dataset.
    /// </summary>
    /// <returns></returns>
    internal static async Task Benchmark_MSMARCO()
    {
        // Benchmark on MS_MARCO 2.1 dataset for question answering.
        // https://huggingface.co/datasets/ms_marco
        using var file = File.OpenRead(@"D:\Dev\PerceptivePyro\datasets\ms_marco\dev_v2.1.json");
        var doc = await JsonDocument.ParseAsync(file);
        var q = doc.RootElement.GetProperty("query").EnumerateObject().OfType<JsonProperty>();
        var p = doc.RootElement.GetProperty("passages");
        ////var a = doc.RootElement.TryGetProperty("answers", out var answers) ? answers : default;
        var tests = from question in q
            let passages = p.GetProperty(question.Name)
                .EnumerateArray()
                .Select(ps =>
                (
                    IsSelected: ps.GetProperty("is_selected").GetInt32() == 1,
                    Text: ps.GetProperty("passage_text").GetString()
                )).ToList()
            ////let answer = a.TryGetProperty(question.Name, out var answer) ? answer.ToString() : null
            select (Question: question.Value.GetString(), Passages: passages);

        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cuda";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        const int TOP_K = 5;
        var scores = 0d;
        var count = 0;
        var results =
            (from test in tests.Take(100)
                let query_embed = output_embeddings(gpt, new[] { test.Question }, "cuda").First()
                let passage_embed = output_embeddings(gpt, test.Passages.Select(p => p.Text).ToList(), "cuda").Zip(test.Passages.Select(p => p.IsSelected), (ab, c) => (ab.Prompt, ab.Embeddings, IsSelected: c ? 1 : 0))
                let r = passage_embed
                    .Select(s => (
                        query_embed.Prompt,
                        s.Prompt,
                        Prediction: 1f - HNSW.Net.CosineDistance.SIMDForUnits(query_embed.Embeddings, s.Embeddings),
                        Actual: s.IsSelected,
                        Loss: (1f - HNSW.Net.CosineDistance.SIMDForUnits(query_embed.Embeddings, s.Embeddings)) - s.IsSelected
                    ))
                    .OrderByDescending(x => x.Prediction)
                    .Take(TOP_K)
                    .ToList()
                select r).ToList();

        foreach (var result in results)
        {
            scores += result.Max(a => a.Actual);
            count++;
        }

        // Last token got us 13% accuracy (top 3)
        // First token got us 15% accuracy (top 3)
        var accuracy = scores / count;
        $"Accuracy for top {TOP_K} {accuracy:P}".Dump();

        foreach (var line in results.Select(a => string.Join("\n", a.Select(b => $"{b.Item1},{b.Item2},{b.Prediction:P},{b.Actual}"))))
        {
            Console.WriteLine(line);
            Console.WriteLine("------");
        }
        /*
        var a = data.Select(d => d.SentenceA).Take(100).ToList();
        var b = data.Select(d => d.SentenceB).Take(100).ToList();
        var true_label = data.Select(d => (d.Relatedness - 1) * .25f).Take(100).ToList(); // Convert 1-5 star rating to percentage


        predictions.Dump();*/
    }

    /// <summary>
    /// Evaluates GPT2 embedding sentence similarity scoring on the SICK dataset.
    /// </summary>
    /// <returns></returns>
    internal static async Task Benchmark_Sick()
    {
        // Benchmark on SICK dataset for sentence similarity score.
        var data = from line in File.ReadLines(@"D:\Dev\PerceptivePyro\datasets\Sick\SICK.txt").Skip(1)
            where !string.IsNullOrWhiteSpace(line)
            let cells = line.Split('\t')
            where cells.Length > 0
            select (Id: cells[0], SentenceA: cells[1], SentenceB: cells[2], Relatedness: float.Parse(cells[4]));

        var a = data.Select(d => d.SentenceA).Take(100).ToList();
        var b = data.Select(d => d.SentenceB).Take(100).ToList();
        var true_label = data.Select(d => (d.Relatedness - 1) * .25f).Take(100).ToList(); // Convert 1-5 star rating to percentage

        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cuda";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        var a_embed = output_embeddings(gpt, a, "cuda");
        var b_embed = output_embeddings(gpt, b, "cuda");
        var predictions = a_embed.Zip(b_embed, true_label)
            .Select(s => (s.First.Prompt, s.Second.Prompt, Prediction: 1f - HNSW.Net.CosineDistance.SIMDForUnits(s.First.Embeddings, s.Second.Embeddings), Actual: s.Third, Loss: (1f - HNSW.Net.CosineDistance.SIMDForUnits(s.First.Embeddings, s.Second.Embeddings)) - s.Third))
            .OrderByDescending(x => x.Loss)
            .ToArray();
        predictions.Dump();
    }

    /// <summary>
    /// Generates embeddings for a set of sentences.
    /// 124M parameter model (gpt2)
    /// </summary>
    /// <returns></returns>
    internal static async Task Gpt2_Embeddings()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cuda";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        // Some text we are giving GPT2 to compare similarity for.
        /*var sentence_data = new[]
        {
            "Hello, I'm a language model.",
            "Hello, I'm a very large language model.",
            "Hi, I'm a language model",
            "Hi, I'm a huge visual model",
            "Hi, I'm a massive speech model",
            "Hi, I'm a massive boat",
            "Hi, I'm a boat",
            "Hello, I'm a visual model",
        };
        */
        var sentence_data = new[]
        {
            "The bright sun rays illuminate the meadow with a warm and comforting light.",
            "I love eating pizza, it's my favorite food in the world.",
            "My favorite hobby is hiking, I enjoy exploring new trails and taking in the scenery.",
            "The concert was amazing, the band played all my favorite songs and the atmosphere was electric.",
        };

        var s = Stopwatch.StartNew();

        // Convert sentences to GPT's semantic embedding representation.
        var sentences = output_embeddings(gpt, sentence_data, "cuda").Dump();

        s.ElapsedMilliseconds.Dump();

        // Convert the query to GPT's semantic embedding representation.
        var query = output_embeddings(gpt, new[] { "The sun is shining brightly, casting a warm glow over the meadow." }, "cuda").Single().Dump();
        var by_similarity = sentences
            .Select(s => (s.Prompt, Difference: HNSW.Net.CosineDistance.SIMDForUnits(s.Embeddings, query.Embeddings)))
            .OrderBy(x => x.Difference)
            .ToArray();
        by_similarity.Select(s => $"Prompt: {s.Prompt}\tDiff: {s.Difference:P}").Dump();
    }

    /// <summary>
    /// Generates embeddings for a set of sentences.
    /// 774M parameter model (gpt2-large)
    /// </summary>
    /// <returns></returns>
    public static async Task Gpt2_Large_Embeddings()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2-large", device);

        // Some text we are giving GPT2 to score similarity to
        var sentence_data = new[]
        {
            "Hello, I'm a language model.",
            "Hello, I'm a very large language model.",
            "Hi, I'm a language model",
            "Hi, I'm a huge visual model",
            "Hi, I'm a massive speech model",
            "Hi, I'm a massive boat",
            "Hi, I'm a boat",
            "Hello, I'm a visual model",
        };

        var sentences = output_embeddings(gpt, sentence_data, "cuda").Dump();
        var query = output_embeddings(gpt, new[] { "Yo, I'm a huge speech model." }, "cuda").Single().Dump();
        var by_similarity = sentences
            .Select(s => (s.Prompt, Difference: HNSW.Net.CosineDistance.SIMD(s.Embeddings, query.Embeddings)))
            .OrderBy(x => x.Difference)
            .ToArray();
        by_similarity.Select(s => $"Prompt: {s.Prompt}\tDiff: {s.Difference:P}").Dump();
    }

    /// <summary>
    /// Generates unconditioned (unprompted) random musings by GPT2.
    /// 124M parameter pre-trained model (gpt2)
    /// </summary>
    /// <returns></returns>
    public static async Task Gpt2_124M_Unconditioned()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT2 pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        // tiktoken style encoding of text into BPE (Byte pair encodings)
        var encoding = GptEncoding.GetEncoding("r50k_base");

        // Start with a single empty token as the starting context, so that GPT2 can get creative with what comes next.
        var gpt_context = torch.zeros(new[] { 1L, 1L }, dtype: torch.@long, device: device);

        // Run the prediction for up to 200 tokens.
        var raw_output = gpt.generate(gpt_context, max_new_tokens: 200);

        // Decode back into human readable form
        encoding.Decode(raw_output[0].data<long>().Select(v => (int)v).ToList()).Dump();
    }

    /// <summary>
    /// Generates unconditioned (unprompted) random musings by GPT2 - 774M parameter pre-trained model (gpt2-large)
    /// </summary>
    /// <returns></returns>
    public static async Task Gpt2_Large_Unconditioned()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT2 pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2-large", device);

        // tiktoken style encoding of text into BPE (Byte pair encodings)
        var encoding = GptEncoding.GetEncoding("r50k_base");

        // Start with a single empty token as the starting context, so that GPT2 can get creative with what comes next.
        var gpt_context = torch.zeros(new[] { 1L, 1L }, dtype: torch.@long, device: device);
        $"Context Shape: {gpt_context.shape.Stringify()}".Dump();

        // Run the prediction for up to 200 tokens.
        var raw_output = gpt.generate(gpt_context, max_new_tokens: 200);

        // Decode back into human readable form
        encoding.Decode(raw_output[0].data<long>().Select(v => (int)v).ToList()).Dump();
    }

    /// <summary>
    /// Generates a prompted response from GPT2 - 124M parameter pre-trained model (gpt2)
    /// </summary>
    /// <returns></returns>
    public static async Task Gpt2_124m_Prompted()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        // Some text we are giving GPT2 to riff on.
        generator(gpt, "Hello, I'm a language model,").ToList().Dump();
    }

    /// <summary>
    /// Generates a prompted response from GPT2 - 774M parameter pre-trained model (gpt2-large)
    /// </summary>
    /// <returns></returns>
    public static async Task Gpt2_Large_Prompted()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2-large", device);

        // Some text we are giving GPT2 to riff on.
        generator(gpt, "Hi, I'm a massive boat").ToList().Dump();
    }

    public static async Task Gpt3_Token_Counts()
    {
        // https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        // p50k_base: Codex models, text-davinci-002, text-davinci-003
        var encoding = GptEncoding.GetEncoding("cl100k_base");

        var prompt = "The quick brown fox jumps over the lazy dog";
        var tokenCount = encoding.Encode(prompt).Count;
        $"{tokenCount} tokens are in {prompt}".Dump();
    }

    public static async Task Gpt4_Token_Counts()
    {
        // https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        // cl100k_base:	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
        var encoding = GptEncoding.GetEncoding("cl100k_base");

        var prompt = "The quick brown fox jumps over the lazy dog";
        var tokenCount = encoding.Encode(prompt).Count;
        $"{tokenCount} tokens are in {prompt}".Dump();
    }

    /// <summary>
    /// Work in Progress - Demonstrates fine tuning GPT2 124M parameter pre-trained model. This one requires a 12GB+ graphics card .
    /// </summary>
    /// <returns></returns>
    public static async Task GPT2_Fine_Tuning()
    {
        // TODO: We're gonna need a bigger GPU...
        var gpt = await GPTModel.from_pretrained("gpt2", "cpu");
        var encoding = GptEncoding.GetEncoding("r50k_base");
        gpt.train();
    }

    /// <summary>
    /// Gets output embeddings from the layer just before the output layer.
    /// This gives "Semantic embeddings" of the whole sentence.
    /// </summary>
    /// <param name="gpt"></param>
    /// <param name="prompt"></param>
    /// <param name="max_length"></param>
    /// <param name="num_return_sequences"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private static IEnumerable<(string Prompt, float[] Embeddings)> output_embeddings(GPTModel gpt, IReadOnlyList<string> prompts, string device = "cpu")
    {
        var encoding = GptEncoding.GetEncoding("r50k_base");
        var encoded_prompt = prompts.Select(p => encoding.Encode(p, new HashSet<string>() { "<|endoftext|>" })).ToList();
        var B = (long)prompts.Count;
        var T = encoded_prompt.Max(e => e.Count);
        var att_mask = torch.zeros(new[] { B, T }, dtype: @long, device: device);
        var gpt_context = torch.zeros(new[] { B, T }, dtype: @long, device: device);

        // TODO: Batchify if a large list of prompts is provided (probably could take IEnumerable if we paginate).
        for (int b = 0; b < encoded_prompt.Count; b++)
        {
            for (int t = 0; t < encoded_prompt[b].Count; t++)
            {
                gpt_context[b, t] = encoded_prompt[b][t];
                att_mask[b, t] = 1f;
            }
        }

        var output = gpt.generate_embedding(gpt_context);
        output.shape.Dump();

        var sentence_embeddings = mean_pooling(output, att_mask); // (B, T, C) -> (B, C)
        sentence_embeddings = sentence_embeddings.normalize(p: 2, dim: 1);
        for (int b = 0; b < encoded_prompt.Count; b++)
        {
            var embeddings = sentence_embeddings[b, ..].data<float>().ToArray(); // (B, T, C) -> (C) // Mean pooled
            yield return (prompts[b], embeddings);
        }

        //for (int b = 0; b < encoded_prompt.Count; b++)
        //{
        //    //var embeddings = output[b, 0, ..].data<float>().ToArray(); // (B, T, C) -> (C) // First token
        //    var embeddings = output[b, encoded_prompt[b].Count - 1, ..].data<float>().ToArray(); // (B, T, C) -> (C) // Last token            
        //    yield return (prompts[b], embeddings);
        //}
    }

    private static Tensor mean_pooling(Tensor model_output, Tensor attention_mask)
    {
        var token_embeddings = model_output; // token embeddings
        var input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).@float();
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min: 1e-9);
    }

    private static IEnumerable<string> generator(GPTModel gpt, string prompt, int max_length = 30, int num_return_sequences = 1, string device = "cpu")
    {
        var encoding = GptEncoding.GetEncoding("r50k_base");
        var encoded_prompt = encoding.Encode(prompt, new HashSet<string>() { "<|endoftext|>" });
        var gpt_context = torch.as_tensor(encoded_prompt, dtype: @long, device: device).reshape(1, encoded_prompt.Count);
        gpt_context.Dump();
        for (int i = 0; i < num_return_sequences; i++)
        {
            var output = gpt.generate(gpt_context, max_new_tokens: max_length, temperature: 0.8d, top_k: 200);
            output.Dump();
            yield return encoding.Decode(output[0].data<long>().Select(v => (int)v).ToList());
        }
    }

    private static void set_seed(int seed)
    {
        torch.manual_seed(seed);
        torch.cuda.manual_seed(seed);
        torch.random.manual_seed(seed);
        torch.backends.cuda.matmul.allow_tf32 = true; // allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = true; // allow tf32 on cudnn
    }
}