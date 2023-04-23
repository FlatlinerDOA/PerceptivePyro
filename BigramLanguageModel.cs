namespace NanoGPTSharp
{
    using TorchSharp;
    using TorchSharp.Modules;
    using F = TorchSharp.torch.nn.functional;

    internal class BigramLanguageModel : torch.nn.Module<torch.Tensor, torch.Tensor?, (torch.Tensor logits, torch.Tensor? loss)>
    {
        private readonly string device;
        private Embedding token_embedding_table;
        private Embedding position_embedding_table;
        private Linear lm_head;

        public BigramLanguageModel(int vocab_size, int n_embd, int block_size, string device) : base(nameof(BigramLanguageModel))
        {
            this.device = device;

            // each token directly reads off the logits for the next token from a lookup table
            this.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd);
            this.position_embedding_table = torch.nn.Embedding(block_size, n_embd);
            this.lm_head = torch.nn.Linear(n_embd, vocab_size); // Layer of indirection from vocab to embeddings
            this.RegisterComponents();
        }

        public override (torch.Tensor, torch.Tensor) forward(torch.Tensor idx, torch.Tensor? targets = null)
        {
            var (B, T) = (idx.shape[0], idx.shape[1]);

            // index and targets are both (b, t) tensor of integers
            var tok_emb = this.token_embedding_table.call(idx); // (Batch, Time, Channel)
            var pos_emb = this.position_embedding_table.call(torch.arange(T, device: this.device));
            var logits = this.lm_head.call(tok_emb); // (Batch, Time, vocab_size)
            
            if (targets is null)
            {
                // NOTE: logits.shape returned here when targets is null is (B,T,C)
                return (logits, null);
            }

            
            var (b, t, c) = (logits.shape[0], logits.shape[1], logits.shape[2]);
            logits = logits.view(b * t, c);
            targets = targets.view(b * t);
            var loss = F.cross_entropy(logits, targets);
            
            // NOTE: logits.shape returned here is (B*T,C)
            return (logits, loss);
        }

        public torch.Tensor generate(torch.Tensor idx, int max_new_tokens)
        {
            // idx is (B, T) array of indices
            foreach (var _ in Enumerable.Range(0, max_new_tokens))
            {
                // get the predictions
                var (logits, __) = this.call(idx, null);
                // focus only on the last time step
                logits = logits[.., -1, ..]; 
                
                ////long lastIndex = logits.shape[0] - 1;
                ////logits = logits.narrow(0, lastIndex, 1);
                
                // apply softmax to get probabilities
                var probs = F.softmax(logits, dim: -1); // (B, C)
                // sample form the distribution
                var idx_next = torch.multinomial(probs, num_samples: 1); // (B, 1)
                // append sampled index to the running sequence
                idx = torch.cat(new[] { idx, idx_next }, dim: 1); // (B, T+1)
            }

            return idx;
        }
    }
}
