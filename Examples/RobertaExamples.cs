using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
            var tokenizer = await RobertaTokenizer.from_pretrained("all-distilroberta-v1");

            var sentences = new[]
            {
                "This is an example sentence",
                "Each sentence is converted"
            };

            foreach (var sentence in sentences)
            {
                var encoded = tokenizer.Encode(sentence);
                var decoded = tokenizer.Decode(encoded);
                $"{sentence} -> {encoded.Stringify()} -> {decoded}".Dump();
            }
        }
    }
}
