namespace NanoGPTSharp.Encodec;

using System.Diagnostics.Contracts;
using System.Numerics;
using System.Runtime.ConstrainedExecution;
using System.Runtime.Intrinsics.X86;
using System.Threading.Channels;
using System.Xml.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = torch.nn.functional;

internal static partial class ConvExtensions
{
    public static nn.Module<Tensor, Tensor> get_activation(this string activation, Dictionary<string, object>? activationParams = null)
    {
        // based on the activation string and its parameters
        nn.Module<Tensor, Tensor> act;
        switch (activation)
        {
            case "GELU":
                act = nn.GELU();
                break;

            case "ReLU":
                act = nn.ReLU();
                break;

            case "LeakyReLU":
                var negativeSlope = (double?)activationParams?.GetValueOrDefault("negative_slope") ?? 0.01d;
                act = nn.LeakyReLU(negativeSlope);
                break;

            case "ELU":
                double alpha = (double?)activationParams?.GetValueOrDefault("alpha") ?? 1d;
                act = nn.ELU(alpha);
                break;

            default:
                throw new ArgumentException($"Activation '{activation}' is not supported.");
        }

        return act;
    }

    public static long get_extra_padding_for_conv1d(this Tensor x, long kernelSize, long stride, long paddingTotal = 0)
    {
        var length = (int)x.shape[2];
        float nFrames = (length - kernelSize + paddingTotal) / (float)stride + 1;
        var idealLength = (int)Math.Ceiling(nFrames) * stride + (kernelSize - paddingTotal);
        return idealLength - length;
    }

    public static Tensor pad1d(this Tensor x, (long Left, long Right) paddings, PaddingModes mode = PaddingModes.Zeros, float value = 0f)
    {
        long length = (long)x.shape[2];
        var (paddingLeft, paddingRight) = paddings;
        if (mode == PaddingModes.Reflect)
        {
            long maxPad = Math.Max(paddingLeft, paddingRight);
            long extraPad = 0;
            if (length <= maxPad)
            {
                extraPad = maxPad - length + 1;
                x = F.pad(x, new long[] { 0, extraPad });
            }

            Tensor padded = F.pad(x, new long[] { paddingLeft, paddingRight }, mode, value);
            long end = (int)padded.shape[2] - extraPad;
            return padded.narrow(2, 0, end);
        }
        else
        {
            return F.pad(x, new long[] { paddingLeft, paddingRight }, mode, value);
        }
    }

    public static nn.Module<Tensor, Tensor> apply_parametrization_norm(this nn.Module<Tensor, Tensor> module, string norm = "none")
    {
        var valid_norms = new HashSet<string> { "none", "weight_norm", "spectral_norm" };
        if (!valid_norms.Contains(norm))
        {
            throw new ArgumentException($"Invalid normalization type: {norm}", nameof(norm));
        }

        if (norm == "weight_norm")
        {
            return weight_norm(module);
        }
        else if (norm == "spectral_norm")
        {
            return spectral_norm(module);
        }
        else
        {
            return module;
        }
    }

    public static nn.Module<Tensor, Tensor> get_norm_module(this nn.Module<Tensor, Tensor> module, bool causal = false, string norm = "none", Dictionary<string, object>? norm_kwargs = null)
    {
        var valid_norms = new HashSet<string> { "none", "weight_norm", "spectral_norm", "layer_norm", "time_group_norm" };
        if (!valid_norms.Contains(norm))
        {
            throw new ArgumentException($"Invalid normalization type: {norm}", nameof(norm));
        }

        if (norm == "layer_norm")
        {
            return new ConvLayerNorm(module.out_channels(), norm_kwargs);
        }
        else if (norm == "time_group_norm")
        {
            if (causal)
            {
                throw new ArgumentException("GroupNorm doesn't support causal evaluation.", nameof(causal));
            }

            return nn.GroupNorm(1, module.out_channels(), (double?)norm_kwargs.GetValueOrDefault("eps") ?? 1E-05d, (bool?)norm_kwargs.GetValueOrDefault("affine") ?? true);
        }
        else
        {
            return nn.Identity();
        }
    }

    public static nn.Module<Tensor, Tensor> weight_norm(this nn.Module<Tensor, Tensor> module, string name = "weight", int dim = 0)
    {

        /*"""Applies weight normalization to a parameter in the given module...math::
         \mathbf{ w} = g \dfrac{\mathbf{ v} }
        {\|\mathbf{ v}\|}

        Weight normalization is a reparameterization that decouples the magnitude
        of a weight tensor from its direction.This replaces the parameter specified
        by: attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
        (e.g. ``'weight_g'``) and one specifying the direction(e.g. ``'weight_v'``).
        Weight normalization is implemented via a hook that recomputes the weight
        tensor from the magnitude and direction before every :meth:`~Module.forward`
        call.

        By default, with ``dim = 0``, the norm is computed independently per output
        channel / plane.To compute a norm over the entire weight tensor, use
        ``dim = None``.

        See https://arxiv.org/abs/1602.07868

        Args:
            module(Module): containing module
            name(str, optional): name of weight parameter
            dim(int, optional): dimension over which to compute the norm

        Returns:
            The original module with the weight norm hook

        Example::

            >>> m = weight_norm(nn.Linear(20, 40), name = 'weight')
            >>> m
            Linear(in_features = 20, out_features = 40, bias = True)
            >>> m.weight_g.size()
            torch.Size([40, 1])
            >>> m.weight_v.size()
            torch.Size([40, 20])

        """*/

        return WeightNorm.Apply(module, name, dim);
    }

    public static Tensor unpad1d(this Tensor x, (long Left, long Right) paddings)
    {
        // Remove padding from x, handling properly zero padding. Only for 1d!
        var (padding_left, padding_right) = paddings;
        Contract.Assert(padding_left >= 0 && padding_right >= 0, $"Invalid padding  ({padding_left}, {padding_right})");
        Contract.Assert((padding_left + padding_right) <= x.shape[^1], "Padding out of bounds");
        var end = x.shape[^1] - padding_right;
        return x[(int)padding_left..(int)end]; // TODO: Verify this is correct.
    }

    public static nn.Module<Tensor, Tensor> spectral_norm(this nn.Module<Tensor, Tensor> module, string name = "weight", int power_iterations = 1, float eps = 1e-12f, int? dim = null)
    {
        throw new NotImplementedException();
    }

    private static int out_channels(this Module module) => module switch
    {
        Conv1d _ => 1,
        Conv2d _ => 2,
        Conv3d _ => 3,
        _ => throw new InvalidOperationException("Must be a ConvNd")
    };
}
