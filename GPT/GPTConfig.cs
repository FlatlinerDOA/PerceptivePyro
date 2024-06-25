using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.GPT
{
    public sealed record GPTConfig
    {
        public int block_size { get; init; } = 1024;
        public int vocab_size { get; init; } = 50304; // GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        public int n_layer { get; init; } = 12;
        public int n_head { get; init; } = 12;
        public int n_embd { get; init; } = 768;
        public double dropout { get; init; } = 0.0d;
        public bool has_bias { get; init; } = true; // True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    }
}
