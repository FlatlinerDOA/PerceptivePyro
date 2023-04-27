namespace NanoGPTSharp.Encodec;

using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;

public class SEANetDecoder : nn.Module<Tensor, Tensor>
{
    private Sequential model;       

    public SEANetDecoder(
        int channels = 1,
        int dimension = 128,
        int n_filters = 32,
        int n_residual_layers = 1,
        List<int>? ratios = null,
        string activation = "ELU",
        Dictionary<string, object>? activation_params = null,
        string? finalActivation = null,
        Dictionary<string, object>? final_activation_params = null,
        string norm = "weight_norm",
        Dictionary<string, object>? norm_params = null,
        int kernelSize = 7,
        int lastKernelSize = 7, 
        int residualKernelSize = 3, 
        int dilationBase = 2,
        bool causal = false,
        PaddingModes pad_mode = PaddingModes.Reflect,
        bool true_skip = false,
        int compress = 2,
        int lstm = 2,
        float trim_right_ratio = 1.0f) : base(nameof(SEANetDecoder))
    {
        // Setup default values for optional arguments
        ratios ??= new List<int> { 8, 5, 4, 2 };
        activation_params ??= new Dictionary<string, object> { { "alpha", 1.0 } };
        norm_params ??= new Dictionary<string, object>();

        var mult = (int)Math.Pow(2, ratios.Count);
        // Initialize the model
        Sequential model = nn.Sequential();
        model.append(new SConv1d(dimension, mult * n_filters, kernelSize, norm: norm, norm_params: norm_params, causal: causal, pad_mode: pad_mode));

        if (lstm > 0)
        {
            model.append(new SLSTM(mult * n_filters, num_layers: lstm));
        }

        for (int i = 0; i < ratios.Count; i++)
        {
            model.append(activation.get_activation(activation_params));
            model.append(new SConvTranspose1d(
                mult * n_filters,
                mult * n_filters / 2,
                kernelSize: i * 2, 
                stride: i,
                norm: norm, 
                norm_params: norm_params,
                causal: causal, 
                trim_right_ratio: trim_right_ratio));

            for (int j = 0; j < n_residual_layers; j++)
            {
                model.append(new SEANetResnetBlock(mult * n_filters / 2, kernel_sizes: new List<int> { residualKernelSize, 1 },
                        dilations: new List<int> { (int)Math.Pow(dilationBase, j), 1 },
                        activation: activation, activation_params: activation_params,
                        norm: norm, norm_params: norm_params, causal: causal,
                        pad_mode: pad_mode, compress: compress, true_skip: true_skip));
            }

            mult /= 2;
        }

        model.append(activation.get_activation(activationParams));
        model.append(new SConv1d(n_filters, channels, lastKernelSize, norm, normParams, causal, padMode));

        if (!string.IsNullOrEmpty(finalActivation))
        {
            model.append(finalActivation.get_activation(finalActivationParams));
        }

        this.model = model;
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return this.model.call(input);
    }
}