namespace NanoGPTSharp.Encodec
{
    internal class NormConv1d : nn.Module<Tensor, Tensor>
    {
        private string norm_type;

        public Conv1d conv;
        public nn.Module<Tensor, Tensor> norm;

        public NormConv1d(long inChannels, long outChannels, long kernelSize, long stride = 1, long padding = 0, long dilation = 1, long groups = 1, bool bias = true, bool causal = false, string norm = "none", Dictionary<string, object>? norm_params = null)
            : base(nameof(NormConv1d))
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Dilation = dilation;
            this.norm_type = norm;
            this.conv = nn.Conv1d(inChannels, outChannels, kernelSize, stride, padding, dilation, groups: groups, bias: bias);
            this.norm = this.conv.apply_parametrization_norm(norm);
            this.norm = this.conv.get_norm_module(causal, norm, norm_params);
            this.RegisterComponents();
        }

        public long KernelSize { get; }
        public long Stride { get; }
        public long Dilation { get; }

        public override Tensor forward(Tensor input)
        {
            var x = this.conv.call(input);
            x = this.norm.call(x);
            return x;
        }
    }
}