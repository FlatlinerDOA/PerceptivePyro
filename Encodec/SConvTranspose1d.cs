using System.Diagnostics.Contracts;
using TorchSharp.Modules;

namespace NanoGPTSharp.Encodec
{
    internal class SConvTranspose1d : nn.Module<Tensor, Tensor>
    {
        private NormConvTranspose1d convtr;
        private bool causal;
        private float trim_right_ratio;

        public SConvTranspose1d(
            int in_channels,
            int out_channels,
            int kernel_size,
            int stride = 1,
            bool causal = false,
            string norm = "none",
            float trim_right_ratio = 1.0f,
            Dictionary<string, object>? norm_kwargs = null) : base(nameof(SConv1d))
        {
            this.convtr = new NormConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal: causal, norm: norm, norm_kwargs: norm_kwargs);
            this.causal = causal;
            this.trim_right_ratio = trim_right_ratio;
            Contract.Assert(this.causal || this.trim_right_ratio == 1f, "`trim_right_ratio` != 1.0 only makes sense for causal convolutions");
            Contract.Assert(this.trim_right_ratio >= 0f && this.trim_right_ratio <= 1.0f);
        }

        public override Tensor forward(Tensor x)
        {
            var kernel_size = this.convtr.KernelSize;
            var stride = this.convtr.Stride;
            var padding_total = kernel_size - stride;

            var y = this.convtr.call(x);

            // We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
            // removed at the very end, when keeping only the right length for the output,
            // as removing it here would require also passing the length at the matching layer
            // in the encoder.
            if (this.causal)
            {
                // Trim the padding on the right according to the specified ratio
                // if trim_right_ratio = 1.0, trim everything from right
                var padding_right = (long)Math.Ceiling(padding_total * this.trim_right_ratio);
                var padding_left = padding_total - padding_right;
                y = y.unpad1d((padding_left, padding_right));
            }
            else
            {
                // Asymmetric padding required for odd strides
                var padding_right = padding_total; // 2
                var padding_left = padding_total - padding_right;
                y = y.unpad1d((padding_left, padding_right));
            }

            return y;
        }
    }
}