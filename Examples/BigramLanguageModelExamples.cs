namespace PerceptivePyro.Examples;

internal class BigramLanguageModelExamples
{
    /// <summary>
    /// how many independent sequences will we process in parallel?
    /// </summary>
    const int batch_size = 8;

    /// <summary>
    /// What is the maximum context length for predictions?
    /// </summary>
    const int block_size = 64;

    /// <summary>
    /// Upper limit on the number of iterations (steps)
    /// </summary>
    const int max_iters = 5000;

    /// <summary>
    /// How many iterations or steps before we evaluate the metrics.
    /// </summary>
    const int eval_interval = 500;

    /// <summary>
    /// Learning rate for gradients.
    /// </summary>
    const double learning_rate = 3e-4;

    /// <summary>
    /// How many steps should be dedicated to evaluation.
    /// </summary>
    const int eval_iters = 200;

    /// <summary>
    /// Number of embeddings
    /// </summary>
    const int n_embd = 64 * 4;

    /// <summary>
    /// Number of attention heads per block
    /// </summary>
    const int n_heads = 4;

    /// <summary>
    /// Number of transformer block layers
    /// </summary>
    const int n_layers = 3;

    /// <summary>
    /// Ratio of dropout for regularization purposes (can help avoid overfitting and smooths out loss curve).
    /// </summary>
    const double dropout = 0.2;

    /// <summary>
    /// Demonstrates training a Transformer Language Model on 1MB of William Shakespeare's works.
    /// </summary>
    /// <returns></returns>
    public static async Task Training_On_Shakespeare()
    {
        var device = torch.cuda_is_available() ? "cuda" : "cpu";

        // Test loading of GPT2
        torch.manual_seed(1337);

        // We always start with a dataset to train on. Let's download the tiny shakespeare dataset
        var filePath = await DownloadDataSetAsync();

        // read it in to inspect it
        var text = await File.ReadAllTextAsync(filePath);
        text.Length.Dump();

        // let's look at the first 1000 characters
        text.Substring(0, 1000).Dump();

        // here are all the unique characters that occur in this text
        var chars = text.ToHashSet().Order().ToArray();
        var vocab_size = chars.Length;
        string.Join("", chars).Dump();
        vocab_size.Dump();

        // create a mapping from characters to integers
        var stoi = chars.Select((c, i) => (c, i)).ToDictionary(_ => _.c, _ => _.i);
        var itos = chars.Select((c, i) => (c, i)).ToDictionary(_ => _.i, _ => _.c);
        var encode = (string s) => s.Select(c => stoi[c]).ToArray();
        var decode = (IEnumerable<int> i) => string.Concat(i.Select(x => itos[x]));

        // let's now encode the entire text dataset and store it into a torch.Tensor
        encode("hii there").Dump();
        decode(encode("hii there")).Dump();

        // let's now encode the entire text dataset and store it into a torch.Tensor
        var data = torch.tensor(encode(text), dtype: torch.@long);
        (data.shape.TotalSize(), data.dtype).Dump();
        data[..1000].Dump();
        var n = (int)(0.9d * data.shape.TotalSize());
        var train_data = data[..n];
        var val_data = data[n..]; // the 1000 characters we looked at earlier will to the GPT look like this
        $"Split: {n}".Dump();

        train_data[..(block_size + 1)].Dump();


        var x = train_data[..block_size];
        var y = train_data[1..(block_size + 1)];

        "Here's how the context is used to predict the target.".Dump();
        foreach (var t in Enumerable.Range(0, block_size))
        {
            var context = x[..(t + 1)];
            var target = y[t];
            $"when input is {context.Stringify()} the target is: {target.Stringify()}".Dump();
        }

        var (xb, yb) = get_batch(train_data, val_data, "train", device);
        $"inputs ({xb.shape.Stringify()}):".Dump();
        xb.Dump();
        $"targets ({yb.shape.Stringify()}):".Dump();
        yb.Dump();
        "-----".Dump();

        "Here's how the context is split into bactches.".Dump();
        foreach (var b in Enumerable.Range(0, batch_size))
        {
            foreach (var t in Enumerable.Range(0, block_size))
            {
                var context = xb[b, ..(t + 1)];
                var target = yb[b, t];
                $"Batch #{b}: When input is {context.tolist().Stringify()} the target is: {target.tolist().Stringify()}".Dump();
            }
        }

        // Setup our bigram language model and copy it to the best device available (pray this is a CUDA GPU).
        var model = new BigramLanguageModel(vocab_size, n_embd, block_size, n_layers, n_heads, dropout, device);
        model = model.to(device);
        var (logits, loss) = model.call(xb, yb);
        logits.shape.Dump();
        loss.Dump();

        // Setup an optimizer with the learning rate
        // NOTE: transformers don't seem to like high learning rates and are affected by the model size!
        // Multiples of 1e-3 or smaller please!
        var optimizer = torch.optim.AdamW(model.parameters(), lr: learning_rate);

        // Our training loop
        foreach (var iter in Enumerable.Range(0, max_iters))
        {
            // Every step that is an even multiple of eval_interval, run an evaluation to see how we're tracking against our validation data set.
            if (iter % eval_interval == 0)
            {
                var est = estimate_loss(model, train_data, val_data, device);
                $"step {iter}: train loss {est["train"]:0.0000}, val loss {est["val"]:0.0000}".Dump();
            }

            var (train_xb, train_yb) = get_batch(train_data, val_data, "train", device);
            var (train_logits, train_loss) = model.call(train_xb, train_yb);
            optimizer.zero_grad();

            // Our loss is never null in training so assert (!) our dominance over the compiler.
            train_loss!.backward();
            optimizer.step();
        }

        var test_context = torch.zeros(new[] { 1L, 1L }, dtype: torch.@long, device: device);
        decode(model.generate(test_context, max_new_tokens: 2000)[0].data<long>().Select(v => (int)v)).Dump();
    }

    static Dictionary<string, float> estimate_loss(BigramLanguageModel model, Tensor train_data, Tensor val_data, string device)
    {
        // Turn off loss gradient calculations while we evaluate.
        using var _ = torch.no_grad();

        // Setup a disposal scope to cleanup tensor memory when we are done with this eval also.
        using var __ = torch.NewDisposeScope();

        // Switch into evaluation mode for Dropout, BatchNorm etc.
        model.eval();

        var output = new Dictionary<string, float>();
        foreach (var split in new[] { "train", "val" })
        {
            var losses = torch.zeros(eval_iters);
            foreach (var k in Enumerable.Range(0, eval_iters))
            {
                var (x, y) = get_batch(train_data, val_data, split, device);
                var (logits, loss) = model.call(x, y);
                losses[k] = loss!.item<float>();
            }

            output[split] = losses.mean().item<float>();
        }

        // Switch back into training mode.
        model.train();
        return output;
    }

    static (Tensor x, Tensor y) get_batch(Tensor train_data, Tensor val_data, string split, string device)
    {
        // generate a small batch of data of inputs x and targets y
        var data = split == "train" ? train_data : val_data;
        var ix = torch.randint((int)data.shape[0] - block_size, new int[] { batch_size });
        var x = torch.stack(ix.data<long>().Select(i => data[(int)i..((int)i + block_size)]));
        var y = torch.stack(ix.data<long>().Select(i => data[((int)i + 1)..((int)i + block_size + 1)]));
        return (x.to(device), y.to(device));
    }

    static Task<string> DownloadDataSetAsync() => DataSets.DownloadDataSetAsync("shakespeare", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
}