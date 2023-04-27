namespace NanoGPTSharp.Encodec;

using TorchSharp.Modules;

class ConvLayerNorm : nn.Module<Tensor, Tensor>
{
    private LayerNorm norm;

    public ConvLayerNorm(long normalized_shape, Dictionary<string, object> norm_kwargs = null) : base(nameof(ConvLayerNorm))
    {
        this.norm = nn.LayerNorm(normalized_shape, (double?)norm_kwargs.GetValueOrDefault("eps") ?? 1E-05d, (bool?)norm_kwargs.GetValueOrDefault("elementwise_affine") ?? true);
    }

    public override Tensor forward(Tensor x)
    {
        // Rearrange 'x' from shape (batch, channels, time) to (batch, time, channels)
        // x = einops.rearrange(x, 'b ... t -> b t ...');
        int numDims = (int)x.Dimensions;
        long[] dimensions = new long[] { 0, numDims - 1 }.Concat(Enumerable.Range(1, numDims - 2).Select(i => (long)i)).ToArray();
        x = x.permute(dimensions); 
        // TODO: Verify this is correct
        

        x = this.norm.forward(x);
        //x = einops.rearrange(x, 'b t ... -> b ... t');
        // Rearrange 'x' back to its original shape (batch, channels, time)
        long[] return_dimensions = new long[] { 0 }.Concat(Enumerable.Range(2, numDims - 2).Select(i => (long)i)).Concat(new long[] { 1 }).ToArray();
        x = x.permute(return_dimensions); // TODO: Verify this is correct
        return x;
    }
}