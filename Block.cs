
namespace NanoGPTSharp
{
    using System;
    using TorchSharp.Modules;
    using static TorchSharp.torch;

    /// <summary>
    /// Transformer block: Communication between tokens followed by computation on tokens.
    /// </summary>
    internal class Block : nn.Module<Tensor, Tensor>
    {
        private int head_size;
        private MultiHeadAttention sa;
        private FeedForward ffwd;
        private LayerNorm ln1;
        private LayerNorm ln2;

        internal Block(int n_embd, int n_heads, int block_size) : base(nameof(Block))
        {
            this.head_size = n_embd / n_heads;
            this.sa = new MultiHeadAttention(n_heads, block_size, n_embd, this.head_size);
            this.ffwd = new FeedForward(n_embd);
            this.ln1 = nn.LayerNorm(n_embd);
            this.ln2 = nn.LayerNorm(n_embd);
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            // Pre-norm
            x = x + this.sa.call(this.ln1.call(x));
            return x + this.ffwd.call(this.ln2.call(x));
        }
    }
}
