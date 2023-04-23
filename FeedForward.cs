namespace NanoGPTSharp
{
    using System;
    using TorchSharp.Modules;
    using static TorchSharp.torch;

    /// <summary>
    /// Multiple heads of self attention, running in parallel.
    /// </summary>
    internal class FeedForward : nn.Module<Tensor, Tensor>
    {
        private Sequential net;

        internal FeedForward(int n_embd) : base(nameof(FeedForward))
        {
            this.net = nn.Sequential(
                nn.Linear(n_embd, n_embd),
                nn.ReLU());
            this.RegisterComponents();
        }

        public override Tensor forward(Tensor input) => this.net.call(input);
    }
}
