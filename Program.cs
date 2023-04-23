namespace NanoGPTSharp
{
    using TorchSharp;
    using static System.Console;
    using static TorchSharp.torch;

    internal class Program
    {
        /// <summary>
        /// What is the maximum context length for predictions?
        /// </summary>
        const int block_size = 8;
        
        /// <summary>
        /// how many independent sequences will we process in parallel?
        /// </summary>
        const int batch_size = 32;
        
        /// <summary>
        /// Upper limit on the number of iterations (steps)
        /// </summary>
        const int max_iters = 30000;
        
        /// <summary>
        /// How many iterations or steps before we evaluate the metrics.
        /// </summary>
        const int eval_interval = 300;
        
        /// <summary>
        /// Learning rate for gradients.
        /// </summary>
        const double learning_rate = 1e-3;
        
        /// <summary>
        /// How many steps should be dedicated to evaluation.
        /// </summary>
        const int eval_iters = 200;

        /// <summary>
        /// Number of embedddings
        /// </summary>
        const int n_embd = 32;

        static async Task Main(string[] args)
        {

            var device = torch.cuda_is_available() ? "cuda" : "cpu";
            $"Running on {device}".Dump();

            torch.manual_seed(1337);

            // We always start with a dataset to train on. Let's download the tiny shakespeare dataset
            await DownloadDataSetAsync();

            self_attention_test();
            return;

            // read it in to inspect it
            var text = await File.ReadAllTextAsync("input.txt");
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
            var val_data = data[n..]; // the 1000 characters we looked at earier will to the GPT look like this
            WriteLine($"Split: {n}");

            train_data[..(block_size + 1)].Dump();


            var x = train_data[..block_size];
            var y = train_data[1..(block_size + 1)];
            foreach (var t in Enumerable.Range(0, block_size))
            {
                var context = x[..(t + 1)];
                var target = y[t];
                $"when input is {context.Stringify()} the target: {target.Stringify()}".Dump();
            }



            var (xb, yb) = get_batch(train_data, val_data, "train", device);
            "inputs:".Dump();
            xb.shape.Dump();
            xb.Dump();
            "targets:".Dump();
            yb.shape.Dump();
            yb.Dump();

            "-----".Dump();

            foreach (var b in Enumerable.Range(0, batch_size))
            {
                foreach (var t in Enumerable.Range(0, block_size))
                {
                    var context = xb[b, ..(t + 1)];
                    var target = yb[b,t];
                    $"when input is {context.tolist().Stringify()} the target: {target.tolist().Stringify()}".Dump();
                }
            }

            var model = new BigramLanguageModel(vocab_size, n_embd, block_size, device);
            model = model.to(device);
            var (logits, loss) = model.call(xb, yb);
            logits.shape.Dump();
            loss.Dump();

            var optimizer = torch.optim.AdamW(model.parameters(), lr: 1e-3);
            foreach (var iter in Enumerable.Range(0, max_iters))
            {
                if (iter % eval_interval == 0)
                {
                    var est = estimate_loss(model, train_data, val_data, device);
                    $"step {iter}: train loss {est["train"]:0.0000}, val loss {est["val"]:0.0000}".Dump();
                }

                ////using var scope = torch.NewDisposeScope();
                var (train_xb, train_yb) = get_batch(train_data, val_data, "train", device);
                var (train_logits, train_loss) = model.call(train_xb, train_yb);
                optimizer.zero_grad();
                train_loss!.backward();
                optimizer.step();
            }

            var gen2 = model.generate(idx: torch.zeros(new[] { 1L, 1 }, dtype: torch.@long, device: device), max_new_tokens: 500);
            decode(gen2[0].data<long>().Select(v => (int)v)).Dump();
        }

        static void self_attention_test()
        {

        }

        static Dictionary<string, float> estimate_loss(BigramLanguageModel model, Tensor train_data, Tensor val_data, string device)
        {
            using var _ = torch.no_grad();
            var output = new Dictionary<string, float>();
            model.eval();
            foreach (var split in new[] { "train", "val" })
            {
                var losses = torch.zeros(eval_iters);
                foreach (var k in Enumerable.Range(0, eval_iters))
                {
                    var (x, y) = get_batch(train_data, val_data, split, device);
                    var (logits, loss) = model.call(x, y);
                    losses[k] = loss.item<float>();
                }

                output[split] = losses.mean().item<float>();
            }

            model.train();
            return output;
        }

        static (Tensor x, Tensor y) get_batch(Tensor train_data, Tensor val_data, string split, string device)
        {
            // generate a small batch of data of inputs x and targets y
            var data = split == "train" ? train_data : val_data;
            var ix = torch.randint((int)data.shape[0] - block_size, new int[] { batch_size });
            var x = torch.stack(ix.data<long>().Select(i => data[(int)i .. ((int)i + block_size)]));
            var y = torch.stack(ix.data<long>().Select(i => data[((int)i + 1)..((int)i + block_size + 1)]));
            return (x.to(device), y.to(device));
        }

        static async Task DownloadDataSetAsync()
        {
            if (File.Exists("input.txt"))
            {
                return;
            }

            var response = await new HttpClient().GetStringAsync("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
            await File.WriteAllTextAsync("input.txt", response);
        }
    }
}