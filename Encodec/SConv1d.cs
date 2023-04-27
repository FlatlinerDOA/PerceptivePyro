namespace NanoGPTSharp.Encodec;

using System;
using F = torch.nn.functional;

public class SConv1d : nn.Module<Tensor, Tensor>
{
    private NormConv1d conv;
    private bool causal;
    private PaddingModes pad_mode;

    public SConv1d(int inChannels, int outChannels, long kernel_size, long stride = 1, long dilation = 1, int groups = 1, bool bias = true, bool causal = false, string norm = "none", Dictionary<string, object>? norm_params = null, PaddingModes pad_mode = PaddingModes.Reflect)
        : base(nameof(SConv1d))
    {
        if (stride > 1 && dilation > 1)
        {
            Console.WriteLine($"SConv1d has been initialized with stride > 1 and dilation > 1 (kernel_size={kernel_size} stride={stride}, dilation={dilation}).");
        }

        conv = new NormConv1d(inChannels, outChannels, kernel_size, stride, dilation: dilation, groups: groups, bias: bias, causal: causal, norm: norm, norm_params: norm_params);
        this.causal = causal;
        this.pad_mode = pad_mode;
    }

    public override Tensor forward(Tensor x)
    {
        var (B, C, T) = (x.shape[0], x.shape[1], x.shape[2]);
        var kernelSize = this.conv.KernelSize;
        var stride = this.conv.Stride;
        var dilation = this.conv.Dilation;
        kernelSize = (kernelSize - 1) * dilation + 1;  // effective kernel size with dilations
        var paddingTotal = kernelSize - stride;
        var extraPadding = x.get_extra_padding_for_conv1d(kernelSize, stride, paddingTotal);

        if (this.causal)
        {
            // Left padding for causal
            x = x.pad1d((paddingTotal, extraPadding), mode: this.pad_mode);
        }
        else
        {
            // Asymmetric padding required for odd strides
            var paddingRight = paddingTotal / 2;
            var paddingLeft = paddingTotal - paddingRight;
            x = x.pad1d((paddingLeft, paddingRight + extraPadding), mode: this.pad_mode);
        }

        return conv.call(x);
    }    
}