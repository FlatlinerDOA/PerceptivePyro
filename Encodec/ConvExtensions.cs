namespace NanoGPTSharp.Encodec;

using F = torch.nn.functional;

internal static class ConvExtensions
{
    public static int GetExtraPaddingForConv1d(this Tensor x, int kernelSize, int stride, int paddingTotal = 0)
    {
        int length = (int)x.shape[2];
        float nFrames = (length - kernelSize + paddingTotal) / (float)stride + 1;
        int idealLength = (int)Math.Ceiling(nFrames) * stride + (kernelSize - paddingTotal);
        return idealLength - length;
    }

    public static Tensor Pad1d(this Tensor x, Tuple<int, int> paddings, PaddingModes mode = PaddingModes.Zeros, float value = 0f)
    {
        int length = (int)x.shape[2];
        int paddingLeft = paddings.Item1;
        int paddingRight = paddings.Item2;
        if (mode == PaddingModes.Reflect)
        {
            int maxPad = Math.Max(paddingLeft, paddingRight);
            int extraPad = 0;
            if (length <= maxPad)
            {
                extraPad = maxPad - length + 1;
                x = F.pad(x, new long[] { 0, extraPad });
            }

            Tensor padded = F.pad(x, new long[] { paddingLeft, paddingRight }, mode, value);
            int end = (int)padded.shape[2] - extraPad;
            return padded.narrow(2, 0, end);
        }
        else
        {
            return F.pad(x, new long[] { paddingLeft, paddingRight }, mode, value);
        }
    }

    public static nn.Module<Tensor, Tensor> ApplyParametrizationNorm(this nn.Module<Tensor, Tensor> module, string norm = "none")
    {
        HashSet<string> CONV_NORMALIZATIONS = new HashSet<string> { "none", "weight_norm", "spectral_norm" };
        if (!CONV_NORMALIZATIONS.Contains(norm))
        {
            throw new ArgumentException($"Invalid normalization type: {norm}");
        }

        if (norm == "weight_norm")
        {
            return WeightNorm(module);
        }
        else if (norm == "spectral_norm")
        {
            return SpectralNorm(module);
        }
        else
        {
            return module;
        }
    }
}
