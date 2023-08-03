using System.Diagnostics;
using System.Diagnostics.Contracts;

namespace PerceptivePyro.Examples;

internal class RobertaExamples
{
    /// <summary>
    /// Tokenizes sentences into numbers with the RobertaTokenizer.
    /// </summary>
    /// <returns></returns>
    public static async Task Roberta_Tokenizing()
    {
        var tokenizer = await RobertaTokenizer.from_pretrained("all-distilroberta-v1");

        var sentences = new[]
        {
            "This is an example sentence",
            "Each sentence is <mask> well",
            "Each sentence is converted <mask>",
            "<mask> sentence is converted well",
            "The bright sun rays illuminate the meadow with a warm and comforting light.",
            "I love eating pizza, it's my favorite food in the world.",
            "My favorite hobby is hiking, I enjoy exploring new trails and taking in the scenery.",
            "The concert was amazing, the band played all my favorite songs and the atmosphere was electric.",
        };

        // These are the outputs from HuggingFace RobertaTokenizer
        var expected_tokens = new[]
        {
            new[] { 0, 713, 16, 41, 1246, 3645, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new[] { 0, 20319, 3645, 16, 50264, 157, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new[] { 0, 20319, 3645, 16, 8417, 50264, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new[] { 0, 50264, 3645, 16, 8417, 157, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            new[] { 0, 133, 4520, 3778, 28496, 38194, 5, 162, 26726, 19, 10, 3279, 8, 29090, 1109, 4, 2, 1, 1, 1 },
            new[] { 0, 100, 657, 4441, 9366, 6, 24, 18, 127, 2674, 689, 11, 5, 232, 4, 2, 1, 1, 1, 1 },
            new[] { 0, 2387, 2674, 21039, 16, 15866, 6, 38, 2254, 8668, 92, 11033, 8, 602, 11, 5, 26815, 4, 2, 1 },
            new[] { 0, 133, 4192, 21, 2770, 6, 5, 1971, 702, 70, 127, 2674, 3686, 8, 5, 5466, 21, 3459, 4, 2 }
        };

        foreach (var (sentence, expected) in sentences.Zip(expected_tokens))
        {
            var encoded = tokenizer.Encode(sentence, new HashSet<string>() { "<mask>" });
            var decoded_expected = tokenizer.Decode(expected);
            var decoded = tokenizer.Decode(encoded);
            Contract.Assert(encoded.SequenceEqual(expected.Take(encoded.Count)), $"Sentence {encoded.Stringify()} was not the expected encoding {expected.Stringify()}");

            $"{sentence} -> {encoded.Stringify()} -> {decoded}".Dump();
        }
    }

    /// <summary>
    /// Demonstrates loading of a RobertaModel from pre-trained weights.
    /// </summary>
    public static async Task Roberta_Loading_Pretrained()
    {
        var model = await RobertaModel.from_pretrained("all-distilroberta-v1");
        var t = SafeTensors.LoadFile(@"./models/all-distilroberta-v1/model.safetensors");
        t.Select(x => "\n" + x.Name + " " + x.Tensor.shape.Stringify()).Dump();
    }

    /// <summary>
    /// Uses all-distilroberta-v1 to create sentence embeddings.
    /// </summary>
    public static async Task Roberta_Sentence_Embeddings()
    {
        var tokenizer = await RobertaTokenizer.from_pretrained("all-distilroberta-v1");
        var model = await RobertaModel.from_pretrained("all-distilroberta-v1");

        var sentences = new[] { "This is an example sentence", "Each sentence is converted" };

        var encoded_input = tokenizer.Tokenize(sentences);
        var embeddings = model.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);

        embeddings.Dump();
    }

    /// <summary>
    /// Uses all-distilroberta-v1 to create sentence embeddings and calculate their similarity.
    /// </summary>
    public static async Task Roberta_Sentence_Similarity()
    {
        var tokenizer = await RobertaTokenizer.from_pretrained("all-distilroberta-v1");
        var model = await RobertaModel.from_pretrained("all-distilroberta-v1");

        /*var sentence_data = new[]
        {
            "The bright sun rays illuminate the meadow with a warm and comforting light.",
            "I love eating pizza, it's my favorite food in the world.",
            "My favorite hobby is hiking, I enjoy exploring new trails and taking in the scenery.",
            "The concert was amazing, the band played all my favorite songs and the atmosphere was electric.", 
        };*/

        /* // Question extraction
        var sentence_data = new[]
        {
            "What feature allows users to see text message replies from customers against an estimate?",
            "How is the reply visibly different from a sent text message?",
            "What happens when a customer replies to a text message for an estimate?",
            "What is created when a customer replies to a text message for an estimate?",
            "What is the due date for the task created when a customer replies to a text message for an estimate?",
            "Who is assigned to the task created when a customer replies to a text message for an estimate?",
            "How is the user notified of the task created when a customer replies to a text message for an estimate?",
            "Can the user mark the text message as dealt with by completing the task?",
            "How is the task linked to the text message in the UI?",
            
            
            "What feature allows Estimator or Customer Service users to send a text message to an owner for an estimate?",
            "Can a text message be sent if no valid mobile phone number is provided?",
            "What happens when a text message is sent to the owner for an estimate?",
            "Is the status of the text message delivery shown on the text message?",
        };*/
        var sentence_data = new[]
        {
            "Introduction to receiving text message replies from customers",
            "Importance of task creation for follow-up",
            "Understanding the differences between sent and received text messages",
            "Creating and assigning tasks with due dates",
            "Alerts and notifications for new tasks",
            "Marking text messages as read/dealt with",
            "Linking tasks and text messages in the UI",
            "Summary and benefits of the tasks feature for tracking follow-up actions",

            "Overview of sending text messages for estimates",
            "Understanding the requirement of a valid mobile phone number",
            "How to verify a mobile phone number for texting",
            "Composing and sending text messages",
            "Best practices for writing text messages for estimates",
            "Monitoring text message delivery status",
            "Troubleshooting text message delivery issues",
            "Summary and benefits of using text messages for estimates.",
        };

        var s = Stopwatch.StartNew();

        // Convert sentences to GPT's semantic embedding representation.
        var sentence_tokens = tokenizer.Tokenize(sentence_data);
        var sentence_embeddings = model.sentence_embeddings(sentence_tokens.input_ids, sentence_tokens.attention_mask).Dump();

        s.ElapsedMilliseconds.Dump();

        // Convert the query to GPT's semantic embedding representation.
        var query_tokens = tokenizer.Tokenize(new[] { "Can iBodyshop send text messages?" });
        var query = model.sentence_embeddings(query_tokens.input_ids, query_tokens.attention_mask).Enumerate2d<float>().Single();
        var by_similarity = sentence_data.Zip(sentence_embeddings.Enumerate2d<float>())
            .Select(s => (s.First, Difference: HNSW.Net.CosineDistance.SIMDForUnits(s.Second, query)))
            .OrderBy(x => x.Difference)
            .ToArray();
        var x = string.Join("\n", by_similarity.Select(s => $"{s.First}\t{s.Difference:P}")).Dump();


        var dates = new[]
        {
            new DateTimeOffset(2023, 05, 29, 12, 48, 34, TimeSpan.FromHours(10)),
            new DateTimeOffset(2023, 05, 29, 11, 48, 34, TimeSpan.FromHours(10)),
            new DateTimeOffset(2023, 05, 28, 12, 48, 34, TimeSpan.FromHours(10)),
        };
        var now = DateTimeOffset.Now;
        var timeDistance = dates.Select(EmbedTime);
    }

    private static float[] EmbedTime(DateTimeOffset dateTime)
    {
        // Year 2000 epoch.
        var age = dateTime - new DateTimeOffset(2000, 1, 1, 0, 0, 0, TimeSpan.Zero);
        var linear_embedding = Enumerable.Repeat(0d, 768 - 6).Concat(new[] { age.TotalDays / 3600.0d / 24.0d / 365.35d, age.TotalDays / 3600.0d / 24.0d, age.TotalHours / 3600.0d, age.TotalMinutes / 60.0d, age.TotalSeconds, 0.0d }).ToArray();

        // L^2 Norm
        return L2Norm(linear_embedding.Select(d => (float)d).ToArray());
    }

    private static float[] L2Norm(float[] input)
    {
        var scale = Math.Sqrt(input.Select(d => Math.Pow(d, 2d)).Sum());
        return input.Select(d => (float)(d / scale)).ToArray();
    }
}