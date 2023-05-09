using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpToken;

namespace NanoGPTSharp.Examples
{
    internal class RobertaExamples
    {
        /// <summary>
        /// Tokenizes sentences into numbers with the RobertaTokenizer.
        /// </summary>
        /// <returns></returns>
        public static async Task Roberta_Tokenizing()
        {
            var gpt = GptEncoding.GetEncoding("r50k_base");
            var tokenizer = await RobertaTokenizer.from_pretrained("all-distilroberta-v1");

            var sentences = new[]
            {
                "This is an example sentence",
                "Each sentence is <mask> converted"
            };

            foreach (var sentence in sentences)
            {
                var encoded = tokenizer.Encode(sentence);
                var decoded = tokenizer.Decode(encoded);
                $"{sentence} -> {encoded.Stringify()} -> {decoded}".Dump();
            }
        }

        /// <summary>
        /// Demonstrates loading of a RobertaModel from pre-trained weights.
        /// </summary>
        public static async Task Roberta_Loading_Pretrained()
        {
            var model = await RobertaModel.from_pretrained("all-distilroberta-v1");
            var t= SafeTensors.LoadFile(@"./models/all-distilroberta-v1/model.safetensors");
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

            var sentence_data = new[]
            {
                "The bright sun rays illuminate the meadow with a warm and comforting light.",
                "I love eating pizza, it's my favorite food in the world.",
                "My favorite hobby is hiking, I enjoy exploring new trails and taking in the scenery.",
                "The concert was amazing, the band played all my favorite songs and the atmosphere was electric.", 
            };

            var s = Stopwatch.StartNew();
        
            // Convert sentences to GPT's semantic embedding representation.
            var sentence_tokens = tokenizer.Tokenize(sentence_data);
            var sentence_embeddings = model.sentence_embeddings(sentence_tokens.input_ids, sentence_tokens.attention_mask).Dump();
        
            s.ElapsedMilliseconds.Dump();

            // Convert the query to GPT's semantic embedding representation.
            var query_tokens = tokenizer.Tokenize(new[] { "The sun is shining brightly, casting a warm glow over the meadow." });
            var query = model.sentence_embeddings(query_tokens.input_ids, query_tokens.attention_mask).Enumerate2d<float>().Single();
            var by_similarity = sentence_data.Zip(sentence_embeddings.Enumerate2d<float>())
                .Select(s => (s.First, Difference: HNSW.Net.CosineDistance.SIMDForUnits(s.Second, query)))
                .OrderBy(x => x.Difference)
                .ToArray();
            by_similarity.Select(s => $"Prompt: {s.First}\tDiff: {s.Difference:P}").Dump();
        }
    }
}
