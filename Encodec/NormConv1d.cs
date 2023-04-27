namespace NanoGPTSharp.Encodec
{
    internal class NormConv1d : nn.Module<Tensor, Tensor>
    {
        private Conv1d conv;
        private nn.Module<Tensor, Tensor> norm;
        private string normType;

        public NormConv1d(long inChannels, long outChannels, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, long groups = 1, bool bias = true, bool causal = false, string norm = "none", Dictionary<string, object> normKwargs = null)
            : base("NormConv1d")
        {
            this.conv = nn.Conv1d(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias);
            this.conv = this.conv.ApplyParametrizationNorm(norm);
            norm = this.conv.GetNormModule(causal, norm, normKwargs);
            normType = norm;
        }

        public override Tensor forward(Tensor input)
        {
            var x = conv.forward(input);
            x = norm.forward(x);
            return x;
        }
    }
}