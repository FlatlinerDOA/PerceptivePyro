namespace PerceptivePyro;

using F = TorchSharp.torch.nn.functional;

internal class BigramLanguageModel : nn.Module<Tensor, Tensor?, (Tensor logits, Tensor? loss)>
{
    private readonly int block_size;
    private readonly string device;
    private Embedding token_embedding_table;
    private Embedding position_embedding_table;
    private Sequential blocks;
    private Linear lm_head;

    public BigramLanguageModel(int vocab_size, int n_embd, int block_size, int n_layer, int n_heads, double dropout, string device) : base(nameof(BigramLanguageModel))
    {
        this.block_size = block_size;
        this.device = device;

        // each token directly reads off the logits for the next token from a lookup table
        this.token_embedding_table = nn.Embedding(vocab_size, n_embd);
        this.position_embedding_table = nn.Embedding(block_size, n_embd);

        var layers = Enumerable.Range(0, n_layer)
            .Select(layer => ("layer_" + layer, (nn.Module<Tensor, Tensor>)new TransformerBlock(n_embd, n_heads: n_heads, block_size, dropout, hasBias: false))) // cast each one so whole list is an enumerable of (string, Module<Tensor, Tensor>)
            .Append(("norm", nn.LayerNorm(n_embd)));
        this.blocks = nn.Sequential(layers);
        this.lm_head = nn.Linear(n_embd, vocab_size); // Layer of indirection from vocab to embeddings

        this.RegisterComponents();
    }

    public override (Tensor, Tensor) forward(Tensor idx, Tensor? targets = null)
    {
        var (B, T) = (idx.shape[0], idx.shape[1]);

        // index and targets are both (b, t) tensor of integers
        var tok_emb = this.token_embedding_table.call(idx); // (Batch, Time, Channel)
        var pos_emb = this.position_embedding_table.call(arange(T, device: this.device));
        var x = tok_emb + pos_emb; // (B, T, C)
        x = this.blocks.call(x); // Apply self-attention. (B, T, C)
        var logits = this.lm_head.call(x); // (Batch, Time, vocab_size)

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

    public Tensor generate(Tensor idx, int max_new_tokens)
    {
        // idx is (B, T) array of indices
        foreach (var _ in Enumerable.Range(0, max_new_tokens))
        {
            // crop idx to the last block_size tokens
            var idx_cond = idx[.., ^this.block_size..];
            // get the predictions
            var (logits, __) = this.call(idx_cond, null);
            // focus only on the last time step
            logits = logits[.., -1, ..];

            ////long lastIndex = logits.shape[0] - 1;
            ////logits = logits.narrow(0, lastIndex, 1);

            // apply softmax to get probabilities
            var probs = F.softmax(logits, dim: -1); // (B, C)
            // sample form the distribution
            var idx_next = multinomial(probs, num_samples: 1); // (B, 1)
            // append sampled index to the running sequence
            idx = cat(new[] { idx, idx_next }, dim: 1); // (B, T+1)
        }

        return idx;
    }
}