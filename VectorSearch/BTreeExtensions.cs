using TorchSharp.Modules;

namespace NanoGPTSharp.VectorSearch
{
    public static class BTreeExtensions
    {
        public static EmbeddingIndex ToBsp(this IEnumerable<float[]> embeddings)
        {
            var tree = new EmbeddingIndex(0);
            foreach (var e in embeddings.Select((e, i) => new IndexedVector(i, e)))
            { 
                tree.Insert(e);
            }

            return tree; // new BSPTree(TreeGen(l, depth + 1), TreeGen(r, depth + 1));
        }

        public static float EuclideanDistance(this IEnumerable<(float First, float Second)> values) => values.Select(lr => lr.First - lr.Second).Norm();

        public static float CosineMargin(this IEnumerable<(float First, float Second)> values) => (float)(values.Sum(lr => (double)lr.First * lr.Second) / (values.Select(v => v.First).Norm() * values.Select(v => v.Second).Norm()));

        public static float EuclideanMargin(this IEnumerable<(float First, float Second)> values, float bias) => bias + (float)values.Sum(lr => (double)lr.First * lr.Second);

        public static float Norm(this IEnumerable<float> a) => MathF.Sqrt(a.Sum(f => f * f));

        public static bool IsZeroVec(this IEnumerable<float> a) => a.All(i => i == 0f);
    }
}
