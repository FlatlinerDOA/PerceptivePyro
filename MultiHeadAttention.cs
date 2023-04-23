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
        private Linear proj;
        private Dropout dropout;

        internal MultiHeadAttention(int num_heads, int block_size, int n_embd, int head_size, double dropout) : base(nameof(Head))
        {
            this.heads = nn.ModuleList<Head>(Enumerable.Range(0, num_heads).Select(h => new Head(block_size, n_embd, head_size, dropout)).ToArray());
            this.proj = nn.Linear(n_embd, n_embd);
            this.dropout = nn.Dropout(dropout);
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            var output = cat(this.heads.Select(h => h.call(input)).ToList(), dim: -1);
            output = this.proj.call(output);
            output = this.dropout.call(output);
            return output;
        }
    }
}
