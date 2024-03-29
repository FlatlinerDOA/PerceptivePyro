﻿namespace PerceptivePyro;

/// <summary>
/// LINQPad style dump extensions for "stringify" nicely and printing out to the console.
/// </summary>
internal static class DumpExtensions
{
    internal static string Stringify(this object? item) => item switch
    {
        null => "<null>",
        string s => s,
        long i => i.ToString(),
        float f => f.ToString("0.000"),
        double d => d.ToString("0.000"),
        ValueTuple v => "(" + ((dynamic)v).Item1.Stringify() + ", " + ((dynamic)v).Item2.Stringify() + ")",
        Tensor t => "tensor(" + t.ToString(TorchSharp.TensorStringStyle.Numpy, "0.0000", 62) + ")",
        Scalar s => s.Stringify(),
        IEnumerable x => x.Stringify(),
        _ => ((object)item)?.ToString() ?? "<null>"
    };

    internal static string Stringify(this IEnumerable items) => "[ "  + string.Join(", ", items.Cast<object>().Select(i => i.Stringify())) + " ]";
 
    internal static string Stringify(this TorchSharp.Scalar item) =>
        item.Type switch
        {
            ScalarType.Float16 => item.ToDouble().ToString(),
            ScalarType.BFloat16 => item.ToDouble().ToString(),
            ScalarType.Float32 => item.ToDouble().ToString(),
            ScalarType.Float64 => item.ToDouble().ToString(),
            ScalarType.ComplexFloat32 => item.ToComplexFloat32().ToString(),
            ScalarType.ComplexFloat64 => item.ToComplexFloat64().ToString(),
            ScalarType.Int16 => item.ToInt16().ToString(),
            ScalarType.Int32 => item.ToInt32().ToString(),
            ScalarType.Int64 => item.ToInt64().ToString(),
            ScalarType.Byte => item.ToByte().ToString(),
            ScalarType.Int8 => item.ToByte().ToString(),
            ScalarType.Bool => item.ToBoolean().ToString(),
            _ => item.ToString() ?? "<null>",
        };

    internal static T Dump<T>(this T item)
    {
        var text = item.Stringify();
        Console.WriteLine(text);
        return item;
    }
}

