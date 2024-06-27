namespace PerceptivePyro.Whisper
{
    using System;
    using System.Diagnostics;
    using TorchSharp.Modules;
    using F = nn.functional;

    public class AudioEncoder : nn.Module<Tensor, Tensor>
    {
        private Conv1d conv1;
        private Conv1d conv2;
        private ModuleList<ResidualAttentionBlock> blocks;
        private LayerNorm ln_post;

        public AudioEncoder(int n_mels, int n_ctx, int n_state, int n_head, int n_layer) : base(nameof(AudioEncoder))
        {
            this.conv1 = nn.Conv1d(n_mels, n_state, kernelSize: 3, padding: 1);
            this.conv2 = nn.Conv1d(n_state, n_state, kernelSize: 3, stride: 2, padding: 1);
            this.register_buffer("positional_embedding", PositionalEmbedding.Sinusoids(n_ctx, n_state));

            this.blocks = nn.ModuleList([..from _ in Enumerable.Range(0, n_layer)
                                           select new ResidualAttentionBlock(n_state, n_head)]);
            this.ln_post = nn.LayerNorm(n_state);
            this.RegisterComponents();
        }

        /// <summary>
        /// Gets the positional_embedding buffer,
        /// which is not to be registered as a weight parameter,
        /// so must be a property not a field.
        /// </summary>
        private Tensor positional_embedding => this.get_buffer("positional_embedding");

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x">The mel spectrogram of the audio. shape = (batch_size, n_mels, n_ctx)</param>
        /// <returns></returns>
        public override Tensor forward(Tensor x)
        {
            x = F.gelu(this.conv1.call(x));
            x = F.gelu(this.conv2.call(x));
            x = x.permute(0, 2, 1);

            var positional_embedding = this.positional_embedding;
            Debug.Assert(x.shape[1..].SequenceEqual(positional_embedding.shape), "incorrect audio shape");
            x = (x + positional_embedding).to(x.dtype);

            foreach (var block in this.blocks)
            {
                x = block.call((x, null, null, null));
            }

            x = this.ln_post.call(x);
            return x;
        }
    }
}
