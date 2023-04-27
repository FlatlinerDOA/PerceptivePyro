namespace NanoGPTSharp.Encodec;

internal class NormConvTranspose1d : nn.Module<Tensor, Tensor>
{
    private nn.Module<Tensor, Tensor> convtr;
    private nn.Module<Tensor, Tensor> norm;
    private string norm_type;

    public NormConvTranspose1d(
        long inputChannel,
        long outputChannel,
        long kernelSize,
        long stride = 1L,
        long padding = 0L,
        long outputPadding = 0L,
        long dilation = 1L,
        PaddingModes paddingMode = PaddingModes.Zeros,
        long groups = 1L,
        bool bias = true,
        Device? device = null,
        ScalarType? dtype = null,
        bool causal = false,
        string norm = "none",
        Dictionary<string, object>? norm_kwargs = null) : base(nameof(NormConvTranspose1d))
    {
        this.convtr = nn.ConvTranspose1d(inputChannel, outputChannel, kernelSize, stride, padding, outputPadding, dilation, paddingMode, groups, bias, device, dtype).apply_parametrization_norm(norm);
        this.norm = this.convtr.get_norm_module(causal, norm, norm_kwargs);
        KernelSize = kernelSize;
        Stride = stride;
        Dilation = dilation;
        this.norm_type = norm;
    }

    public long KernelSize { get; }
    public long Stride { get; }
    public long Dilation { get; }

    public override Tensor forward(Tensor x)
    {
        x = this.convtr.call(x);
        x = this.norm.call(x);
        return x;
    }
}
