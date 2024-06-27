using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.Whisper.Decoding
{
    /// <summary>
    /// Union[str, List[int]]
    /// </summary>
    /// <param name="text"></param>
    /// <param name="tokens"></param>
    public record class Prompt(string? text = null, List<int>? tokens = null)
    {
        public static implicit operator string?(Prompt prompt) => prompt.text;
        public static implicit operator List<int>?(Prompt prompt) => prompt.tokens;
        public static implicit operator Prompt(string text) => new Prompt(text);
        public static implicit operator Prompt(List<int> tokens) => new Prompt(null, tokens);
    }
}
