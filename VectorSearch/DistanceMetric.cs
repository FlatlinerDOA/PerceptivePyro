namespace NanoGPTSharp
{
    public enum DistanceMetric
    {
        /// <summary>
        /// Default metric.
        /// </summary>
        Default = 0,
        
        /// <summary>
        /// Euclidean distance metric.
        /// </summary>
        Euclidean = 1,

        /// <summary>
        /// Angular or cosine distance metric.
        /// </summary>
        Cosine = 2
    }
}