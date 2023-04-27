namespace NanoGPTSharp.Encodec
{
    internal class SConvTranspose1d
    {
        private object value1;
        private object value2;
        private object kernelSize;
        private object stride;
        private string norm;
        private Dictionary<string, object> normParams;
        private bool causal;
        private float trimRightRatio;

        public SConvTranspose1d(object value1, object value2, object kernelSize, object stride, string norm, Dictionary<string, object> normParams, bool causal, float trimRightRatio)
        {
            this.value1 = value1;
            this.value2 = value2;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.norm = norm;
            this.normParams = normParams;
            this.causal = causal;
            this.trimRightRatio = trimRightRatio;
        }
    }
}