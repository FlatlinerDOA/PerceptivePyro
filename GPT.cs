namespace NanoGPTSharp;

using System;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using F = TorchSharp.torch.nn.functional;

internal class GPT : nn.Module<Tensor, Tensor?, (Tensor logits, Tensor loss)>
{
    private GPTConfig config;
    private Embedding wte;
    private Embedding wpe;
    private Dropout drop;
    private LayerNorm ln_f;
    /// <summary>
    /// Attention heads
    /// </summary>
    private Block[] h;
    private ModuleDict<nn.Module> transformer;
    private Linear lm_head;

    internal GPT(GPTConfig config) : base(nameof(GPT))
    {
        Contract.Assert(config.vocab_size is not 0);
        Contract.Assert(config.block_size is not 0);

        this.config = config;

        this.wte = nn.Embedding(config.vocab_size, config.n_embd);
        this.wpe = nn.Embedding(config.block_size, config.n_embd);
        this.drop = nn.Dropout(config.dropout);
        this.ln_f = new LayerNorm(config.n_embd, hasBias: config.has_bias);
        this.h = (from _ in Enumerable.Range(0, config.n_layer) select new Block(config)).ToArray();
        this.transformer = nn.ModuleDict(
            ("wte", this.wte),
            ("wpe", this.wpe),
            ("drop", this.drop),
            ("h", nn.ModuleList(this.h)),
            ("ln_f", this.ln_f)
        );
        this.lm_head = nn.Linear(config.n_embd, config.vocab_size, hasBias: false);

        // with weight tying when using torch.compile() some warnings get generated:
        // "UserWarning: functional_call was passed multiple values for tied weights.
        // This behavior is deprecated and will be an error in future versions"
        // not 100% sure what this is, so far seems to be harmless.
        // TODO: investigate
        this.wte.weight = this.lm_head.weight; // https://paperswithcode.com/method/weight-tying

        // init all weights
        this.apply(this._init_weights);
        // apply special scaled init to the residual projections, per GPT-2 paper
        foreach (var (pn, p) in this.named_parameters())
        {
            if (pn.EndsWith("c_proj.weight"))
            {
                torch.nn.init.normal_(p, mean: 0.0, std: 0.02 / Math.Sqrt(2 * config.n_layer));
            }
        }

        // report number of parameters
        Debug.WriteLine($"number of parameters: {this.get_num_params() / 1e6}M");
        this.RegisterComponents();
    }

    public static GPT from_pretrained(string model_type, GPTConfig override_args = null)
    {
        Debug.WriteLine($"loading weights from pretrained: {model_type}");

        // n_layer, n_head and n_embd are determined from model_type
        var model_configs = new Dictionary<string, GPTConfig>()
        {
            ["gpt2"] = new() { n_layer = 12, n_head = 12, n_embd = 768 },  // 124M params
            ["gpt2-medium"] = new() { n_layer = 24, n_head = 16, n_embd = 1024 }, // 350M params
            ["gpt2-large"] = new() { n_layer = 36, n_head = 20, n_embd = 1280 }, // 774M params
            ["gpt2-xl"] = new() { n_layer = 48, n_head = 25, n_embd = 1600 }, // 1558M params
        };

        Contract.Assert(model_configs.ContainsKey(model_type), $"Invalid model_type: {model_type}");
        var config = model_configs[model_type];

        Debug.WriteLine("forcing vocab_size=50257, block_size=1024, bias=True");
        config = config with
        {
            vocab_size = 50257, // always 50257 for GPT model checkpoints
            block_size = 1024, // always 1024 for GPT model checkpoints
            has_bias = true, // always True for GPT model checkpoints,
            dropout = override_args?.dropout ?? config.dropout, // overriding dropout rate
        };

        // create a from-scratch initialized minGPT model
        var model = new GPT(config);
        var sd = model.state_dict();
        var sd_keys = (from k in sd.Keys where !k.EndsWith(".attn.bias") select k); // discard this mask / buffer, not a param


        /* TODO: Convert
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
        */
    }

    public override (Tensor, Tensor?) forward(Tensor idx, Tensor? targets = null)
    {
        var device = idx.device;
        var (b, t) = (idx.size()[0], idx.size()[1]);
        Contract.Assert(t <= this.config.block_size, $"Cannot forward sequence of length {t}, block size is only {this.config.block_size}");
        var pos = torch.arange(0, t, dtype: torch.@long, device: device).unsqueeze(0); // shape (1, t)

        // forward the GPT model itself
        var tok_emb = this.wte.call(idx); // token embeddings of shape (b, t, n_embd)
        var pos_emb = this.wpe.call(pos); // position embeddings of shape (1, t, n_embd)
        var x = this.drop.call(tok_emb + pos_emb);
        foreach (var block in this.h)
        {
            x = block.call(x);
            x = this.ln_f.call(x);
        }

        Tensor logits;
        Tensor? loss;
        if (targets is not null)
        {
            //if we are given some desired targets also calculate the loss
            logits = this.lm_head.call(x);
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index: -1);
        }
        else
        {
            // inference-time mini-optimization: only forward the lm_head on the very last position
            long lastIndex = x.shape[1] - 1;
            var x_narrowed = x.narrow(1, lastIndex, 1); // Equivalent to x[:, [-1], :]
            logits = lm_head.forward(x_narrowed); // note: using list [-1] to preserve the time dim
            loss = null;
        }

        return (logits, loss);
    }

    /// <summary>
    /// Return the number of parameters in the model.
    /// For non-embedding count(default), the position embeddings get subtracted.
    /// The token embeddings would too, except due to the parameter sharing these
    /// params are actually used as weights in the final layer, so we include them.
    /// </summary>
    /// <param name="non_embedding"></param>
    /// <returns></returns>
    public long get_num_params(bool non_embedding = true) 
    {
        var n_params = (from p in this.parameters()
                       select p.numel()).Sum();
        if (non_embedding)
        {
            n_params -= this.wpe.weight?.numel() ?? 0;
        }

        return n_params;
    }

    /// <summary>
    /// model surgery to decrease the block size if necessary
    /// e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    /// but want to use a smaller block size for some smaller, simpler model
    /// </summary>
    /// <param name="new_block_size">New desired block size</param>
    /// <returns></returns>
    public void crop_block_size(int new_block_size)
    {
        Contract.Assert(new_block_size <= this.config.block_size);
        this.config = this.config with { block_size = new_block_size };
        this.wpe.weight = nn.Parameter(this.wpe.weight[..new_block_size]);
        foreach (var block in this.h)
        {
            var bias = block.attn.bias;
            if (bias is not null)
            {
                block.attn.bias = bias[.., .., ..new_block_size, ..new_block_size];
            }
        }
    }

    private void _init_weights(nn.Module module)
    {
        if (module is Linear lin)
        {
            nn.init.normal_(lin.weight, mean: 0.0, std: 0.02);
            if (lin.bias is not null)
            {
                nn.init.zeros_(lin.bias);
            }
        }
        else if (module is Embedding e)
        {
            nn.init.normal_(e.weight, mean: 0.0, std: 0.02);
        }
    }
}
