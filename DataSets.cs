namespace PerceptivePyro;

/// <summary>
/// Provides simplified access to data sets by downloading them first.
/// </summary>
public static class DataSets
{
    /// <summary>
    /// Attempts to download a data set or just returns it's file path if it has already been downloaded.
    /// </summary>
    /// <param name="dataset">The name of the dataset.</param>
    /// <param name="url">The url to download the data set from if it isn't already.</param>
    /// <param name="cancellation">Cancels the download.</param>
    /// <returns>The file path to the downloaded data set.</returns>
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
