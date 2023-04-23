namespace NanoGPTSharp
{
    using System;
    using TorchSharp.Modules;
    using static TorchSharp.torch;

    /// <summary>
    /// Multiple heads of self attention, running in parallel.
    /// </summary>
    internal class MultiHeadAttention : nn.Module<Tensor, Tensor>
    {
        private ModuleList<Head> heads;

        internal MultiHeadAttention(int num_heads, int block_size, int n_embd, int head_size) : base(nameof(Head))
        {
            this.heads = nn.ModuleList<Head>(Enumerable.Range(0, num_heads).Select(h => new Head(block_size, n_embd, head_size)).ToArray());
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor input) => cat(this.heads.Select(h => h.call(input)).ToList(), dim: -1);
    }
}
