namespace NanoGPTSharp.Encodec;

using System;
using F = torch.nn.functional;

public class SConv1d : nn.Module<Tensor, Tensor>
{
    private NormConv1d conv;
    private bool causal;
    private PaddingModes padMode;

    public SConv1d(int inChannels, int outChannels, int kernelSize, int stride = 1, int dilation = 1, int groups = 1, bool bias = true, bool causal = false, string norm = "none", Dictionary<string, object> normParams = null, PaddingModes padMode = PaddingModes.Reflect)
        : base("SConv1d")
    {
        if (stride > 1 && dilation > 1)
        {
            Console.WriteLine($"SConv1d has been initialized with stride > 1 and dilation > 1 (kernel_size={kernelSize} stride={stride}, dilation={dilation}).");
        }

        conv = new NormConv1d(inChannels, outChannels, kernelSize, stride, dilation: dilation, groups: groups, bias: bias, causal: causal, norm: norm, normParams: normParams);
        this.causal = causal;
        this.padMode = padMode;
    }

    public override Tensor forward(Tensor x)
    {
        var (B, C, T) = (x.shape[0], x.shape[1], x.shape[2]);
        int kernelSize = conv.Conv.kernel_size[0];
        int stride = conv.Conv.stride[0];
        int dilation = conv.Conv.dilation[0];
        kernelSize = (kernelSize - 1) * dilation + 1;  // effective kernel size with dilations
        int paddingTotal = kernelSize - stride;
        int extraPadding = GetExtraPaddingForConv1d(x, kernelSize, stride, paddingTotal);

        if (causal)
        {
            // Left padding for causal
            x = Pad1d(x, (paddingTotal, extraPadding), mode: padMode);
        }
        else
        {
            // Asymmetric padding required for odd strides
            int paddingRight = paddingTotal / 2;
            int paddingLeft = paddingTotal - paddingRight;
            x = Pad1d(x, (paddingLeft, paddingRight + extraPadding), mode: padMode);
        }

        return conv.call(x);
    }

    // TODO: Implement GetExtraPaddingForConv1d and Pad1d methods


    
}