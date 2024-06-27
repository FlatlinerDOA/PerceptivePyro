namespace PerceptivePyro.Whisper;

using PerceptivePyro.Whisper.Decoding;
using System;
using System.IO.Compression;
using System.Text;
using TorchSharp.Modules;

using CacheHookRemover = TorchSharp.torch.nn.HookableModule<Func<nn.Module<Tensor, Tensor>, Tensor, Tensor>, Func<nn.Module<Tensor, Tensor>, Tensor, Tensor, Tensor>>.HookRemover;

public class WhisperModel : nn.Module<(Tensor mel, Tensor tokens), Tensor>
{
    public ModelDimensions dims;
    public AudioEncoder encoder;
    public TextDecoder decoder;
    public Tensor all_heads;

    public WhisperModel(ModelDimensions dims) : base(nameof(WhisperModel))
    {
        this.dims = dims;
        this.encoder = new AudioEncoder(
        this.dims.n_mels,
        this.dims.n_audio_ctx,
        this.dims.n_audio_state,
        this.dims.n_audio_head,
        this.dims.n_audio_layer
    );
        this.decoder = new TextDecoder(
            this.dims.n_vocab,
            this.dims.n_text_ctx,
            this.dims.n_text_state,
            this.dims.n_text_head,
            this.dims.n_text_layer
        );

        // use the last half among the decoder layers for time alignment by public object ault;
        // to use a specific set of heads, see `set_alignment_heads()` below.
        this.all_heads = torch.zeros(this.dims.n_text_layer, this.dims.n_text_head, dtype: torch.@bool);
        this.all_heads[(this.dims.n_text_layer / 2)..] = true;
        this.register_buffer("alignment_heads", this.all_heads.to_sparse(), persistent: false);
    }

    public static byte[] Base85Decode(string input) => Encoding.ASCII.GetBytes(input)
        .Select(x => (byte)(x - 33))
        .ToArray();

    public static byte[] Decompress(byte[] data)
    {
        using (var compressedStream = new MemoryStream(data))
        using (var decompressedStream = new MemoryStream())
        using (var gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress))
        {
            gzipStream.CopyTo(decompressedStream);
            return decompressedStream.ToArray();
        }
    }

    private void set_alignment_heads(string dump)
    {
        // Decode the base85 string
        var decodedBytes = Base85Decode(dump);

        // Decompress the gzip data
        var decompressedBytes = Decompress(decodedBytes);

        // Convert the byte array to a boolean array
        bool[] boolArray = decompressedBytes.Select(b => b != 0).ToArray();

        // Convert the boolean array to a Torch tensor
        var array = torch.tensor(boolArray, dtype: torch.@bool);
        var mask = array.reshape(this.dims.n_text_layer, this.dims.n_text_head);
        this.register_buffer("alignment_heads", mask.to_sparse(), persistent: false);
    }

    public Tensor embed_audio(Tensor mel) => this.encoder.call(mel);

    public Tensor logits(Tensor tokens, Tensor audio_features) => this.decoder.call((tokens, audio_features, null));

    public override Tensor forward((Tensor mel, Tensor tokens) input) => this.forward(input.mel, input.tokens);

    public Tensor forward(Tensor mel, Tensor tokens) => this.decoder.call((tokens, this.encoder.call(mel), null));

    public Device device => this.parameters().First().device;

    public bool is_multilingual => this.dims.n_vocab >= 51865;

    public int num_languages => this.dims.n_vocab - 51765 - (this.is_multilingual ? 1 : 0);

    public (Dictionary<nn.Module<Tensor, Tensor>, Tensor> cache, List<CacheHookRemover> hooks) install_kv_cache_hooks(Dictionary<nn.Module<Tensor, Tensor>, Tensor> cache = null)
    {
        /*
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        */
        cache = cache is not null ? new(cache) : new();
        var hooks = new List<CacheHookRemover>();

        Tensor save_to_cache(nn.Module<Tensor, Tensor> module, Tensor _, Tensor output)
        {
            if (!cache.ContainsKey(module) || output.shape[1] > this.dims.n_text_ctx)
            {
                // save as-is, for the first token or cross attention
                cache[module] = output;
            }
            else
            {
                cache[module] = torch.cat([cache[module], output], dim: 1).detach();
            }
            return cache[module];
        }

        void install_hooks(nn.Module layer)
        {
            if (layer is MultiHeadAttention head)
            {
                hooks.Add(head.key.register_forward_hook(save_to_cache));
                hooks.Add(head.value.register_forward_hook(save_to_cache));
            }

        }

        this.decoder.apply(install_hooks);
        return (cache, hooks);
    }

    public (Tensor lang_tokens, List<Dictionary<string, float>> lang_probs) detect_language(Tensor audio_features, WhisperTokenizer tokenizer) => throw new NotImplementedException();  //Decoding.detect_language_function();
    
    public object transcribe() => throw new NotImplementedException(); // Decoding.transcribe_function();


    public DecodingResult decode(Tensor mel, DecodingOptions? options = null, Dictionary<string, object>? kwargs = null) => decode_batch(mel, options, kwargs).First();

    public IEnumerable<DecodingResult> decode_batch(Tensor mel, DecodingOptions? options = null, Dictionary<string, object>? kwargs = null)
    {
        using var _ = no_grad();

        var single = mel.ndim == 2;
        if (single)
        {
            mel = mel.unsqueeze(0);
        }

        if (kwargs is not null)
        {
            options = options.With(kwargs);
        }

        var result = new DecodingTask(this, options).Run(mel);
        return single ? result.Take(1) : result;
    }
}
