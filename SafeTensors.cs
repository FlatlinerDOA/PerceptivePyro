namespace NanoGPTSharp;

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using System.Diagnostics.Contracts;
using System.Data.Common;
using static System.Runtime.InteropServices.JavaScript.JSType;

/// <summary>
/// .NET 
/// </summary>
internal static class SafeTensors
{
    public static Dictionary<string, ScalarType> dtypes = new Dictionary<string, ScalarType>
    {
        ["F16"] = torch.float16,
        ["F32"] = torch.float32,
        ["I8"] = torch.int8,
        ["I16"] = torch.int16,
        ["I32"] = torch.int32,
        ["I64"] = torch.int64,
        ["U8"] = torch.uint8
    };

    /// <summary>
    /// Loads a .safetensors file into a sequence of named Tensors
    /// </summary>
    /// <param name="filename"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static IEnumerable<(string Name, Tensor Tensor)> LoadFile(string filename, string device)
    {
        using var fs = new FileStream(filename, FileMode.Open, FileAccess.Read);

        var maxLength = fs.Length;
        using var reader = new BinaryReader(fs);

        var metadataLength = (int)reader.ReadInt64();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = Encoding.UTF8.GetString(metadataBytes);

        Contract.Assert(metadataJson != null, "Invalid .safetensors file, metadata is null");
        var metadata = JsonSerializer.Deserialize<Dictionary<string, SafeTensorMetadata>>(metadataJson);

        var result = new Dictionary<string, Tensor>();
        
        // Read them out in order, for file system cache's sake.
        foreach (var item in metadata.OrderBy(m => m.Value.data_offsets?[0] ?? 0))
        {
            if (item.Key != "__metadata__")
            {
                yield return (item.Key, CreateTensor(reader, item.Value, metadataLength + sizeof(Int64), device));
            }
        }
    }

    public static IEnumerable<(string Name, SafeTensorMetadata Metadata)> DescribeFile(string filename)
    {
        using var fs = new FileStream(filename, FileMode.Open, FileAccess.Read);

        var maxLength = fs.Length;
        using var reader = new BinaryReader(fs);

        var metadataLength = (int)reader.ReadInt64();
        var metadataBytes = reader.ReadBytes(metadataLength);
        var metadataJson = Encoding.UTF8.GetString(metadataBytes);

        Contract.Assert(metadataJson != null, "Invalid .safetensors file, metadata is null");
        var metadata = JsonSerializer.Deserialize<Dictionary<string, SafeTensorMetadata>>(metadataJson);
        return metadata.OrderBy(m => m.Value.data_offsets?[0] ?? 0).Select(m => (m.Key, m.Value));
    }

    public static Tensor CreateTensor(BinaryReader reader, SafeTensorMetadata info, int offset, string device)
    {
        var dtype = dtypes[info.dtype];
        var shape = info.shape;
        var (start, stop) = (info.data_offsets[0], info.data_offsets[1]);

        var length = (int)(stop - start);
        reader.BaseStream.Seek(offset + start, SeekOrigin.Begin);

        Span<byte> b = reader.ReadBytes(length).AsSpan();

        // HACK: Load into an array first, doesn't avoid a copy, but just wanted to get it working. Limits tensors to 2gb or less.
        Tensor output = torch.empty(shape, dtype, torch.CPU);
        output.bytes = b;
        output.to(device);

        return output;
    }

    public sealed record SafeTensorMetadata
    {
        public string dtype { get; set; }

        public long[] shape { get; set; }

        public long[] data_offsets { get; set; }
    }
}
