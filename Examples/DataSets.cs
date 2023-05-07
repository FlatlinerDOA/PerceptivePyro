namespace NanoGPTSharp.Examples;

internal static class DataSets
{
    public static async Task<string> DownloadDataSetAsync(string dataset, string url, CancellationToken cancellation = default)
    {
        var fileName = Path.GetFileName(url);
        var filePath = Path.GetFullPath($@".\datasets\{dataset}\{fileName}");
        if (File.Exists(filePath))
        {
            return filePath;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
        $"Downloading {dataset} to {filePath}".Dump();
        var stream = await new HttpClient().GetStreamAsync(url);
        using var outputStream = File.OpenWrite(filePath);
        await stream.CopyToAsync(outputStream, cancellation);
        return filePath;
    }
}
