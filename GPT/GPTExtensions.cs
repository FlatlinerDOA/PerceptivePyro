using SharpToken;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.GPT
{
    public static class GPTExtensions
    {
        public static IEnumerable<string> generator(this GPTModel gpt, string prompt, int max_length = 30, int num_return_sequences = 1, string device = "cpu")
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
    }
}
