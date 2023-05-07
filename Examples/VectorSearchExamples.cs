using NanoGPTSharp.VectorSearch;

namespace NanoGPTSharp.Examples;

public static class VectorSearchExamples
{
    public static async Task VectorImageSearch()
    {
        var filePath = await DataSets.DownloadDataSetAsync("deep1b", "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.10M.fbin");
        var image_embeddings = from image in ReadFBin(filePath)
                               select image;

        const int TestSize = 500000;

        // Step 1 - Load the entire database into BSPTree
        var bspTree = image_embeddings.Take(TestSize).ToBsp();

        // Step 2 - Query
        var queryFilePath = await DataSets.DownloadDataSetAsync("deep1b", "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin");
        var groundtruthFilePath = await DataSets.DownloadDataSetAsync("deep1b", "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/groundtruth.public.10K.ibin");
        var test_embeddings = from qg in ReadFBin(queryFilePath).Zip(ReadIBin(groundtruthFilePath))
                               let test_query = qg.First
                               let expected_id = qg.Second
                               select (test_query, expected_id);

        const int TestK = 1;
        foreach (var (test_query, expected_id) in test_embeddings.Where(t => t.expected_id.First() < TestSize))
        {
            var result = bspTree.SearchKNN(test_query, TestK);
            var correct = result.Select(i => i.Id).SequenceEqual(expected_id.Take(TestK));
            Console.Write(correct ? "o " : "x ");
        }
    }

    /// <summary>
    /// Loads .fbin files from: <a href="https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search">Benchmarks for Billion-Scale Similarity Search</a>
    /// </summary>
    private static IEnumerable<float[]> ReadFBin(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        var vectors = reader.ReadUInt32();
        var dimensions = reader.ReadUInt32();

        (vectors, dimensions).Dump();
        for (long i = 0; i < vectors; i++)
        {
            var buffer = new float[dimensions];
            for (int d = 0; d < dimensions; d++)
            {
                buffer[d] = reader.ReadSingle();
            }

            yield return buffer;
        }
    }

    /// <summary>
    /// Loads .fbin files from: <a href="https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search">Benchmarks for Billion-Scale Similarity Search</a>
    /// </summary>
    private static IEnumerable<int[]> ReadIBin(string filePath)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(fs);
        var vectors = reader.ReadUInt32();
        var dimensions = reader.ReadUInt32();

        (vectors, dimensions).Dump();
        for (long i = 0; i < vectors; i++)
        {
            var buffer = new int[dimensions];
            for (int d = 0; d < dimensions; d++)
            {
                buffer[d] = reader.ReadInt32();
            }

            yield return buffer;
        }
    }
}
