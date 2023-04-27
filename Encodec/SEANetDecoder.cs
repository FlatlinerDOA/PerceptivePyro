namespace NanoGPTSharp.Encodec;

using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;

public class SEANetDecoder : nn.Module<Tensor, Tensor>
{
    private Sequential model;
    private List<int> ratios;
    private Dictionary<string, object> activationParams;
    private Dictionary<string, object> normParams;

    public SEANetDecoder(
        int channels = 1,
        int dimension = 128,
        int nFilters = 32,
        int nResidualLayers = 1,
        List<int>? ratios = null,
        string activation = "ELU",
        Dictionary<string, object>? activationParams = null,
        string? finalActivation = null,
        Dictionary<string, object>? finalActivationParams = null,
        string norm = "weight_norm",
        Dictionary<string, object>? normParams = null,
        int kernelSize = 7,
        int lastKernelSize = 7, 
        int residualKernelSize = 3, 
        int dilationBase = 2,
        bool causal = false,
        PaddingModes padMode = PaddingModes.Reflect,
        bool trueSkip = false,
        int compress = 2,
        int lstm = 2,
        float trimRightRatio = 1.0f) : base(nameof(SEANetDecoder))
    {
        // Setup default values for optional arguments
        this.ratios ??= new List<int> { 8, 5, 4, 2 };
        this.activationParams ??= new Dictionary<string, object> { { "alpha", 1.0 } };
        this.normParams ??= new Dictionary<string, object>();

        var mult = (int)(2 ** this.ratios.Count);
        // Initialize the model
        Sequential model = nn.Sequential();
        model.append(new SConv1d(dimension, mult * nFilters, kernelSize, this.norm, this.normParams, causal, padMode));

        if (lstm > 0)
        {
            model.append(new SLSTM(mult * nFilters, num_layers: lstm));
        }

        for (int i = 0; i < ratios.Count; i++)
        {
            model.append(GetActivation(activation, activationParams));
            model.append(new SConvTranspose1d(mult * nFilters, mult * nFilters / 2,
                    kernelSize: ratio * 2, stride: ratio,
                    norm: norm, normParams: normParams,
                    causal: causal, trimRightRatio: trimRightRatio));

            for (int j = 0; j < nResidualLayers; j++)
            {
                model.append(new SEANetResnetBlock(mult * nFilters / 2, kernelSizes: new List<int> { residualKernelSize, 1 },
                        dilations: new List<int> { (int)Math.Pow(dilationBase, j), 1 },
                        activation: activation, activationParams: activationParams,
                        norm: norm, normParams: normParams, causal: causal,
                        padMode: padMode, compress: compress, trueSkip: trueSkip));
            }

            mult /= 2;
        }

        model.append(GetActivation(activation, activationParams));
        model.append(new SConv1d(nFilters, channels, lastKernelSize, norm, normParams, causal, padMode));

        if (!string.IsNullOrEmpty(finalActivation))
        {
            model.append(GetActivation(finalActivation, finalActivationParams));
        }

        this.model = model;
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return this.model.call(input);
    }

    private nn.Module<Tensor, Tensor> GetActivation(string activation, Dictionary<string, object> activationParams)
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
                var negativeSlope = (double?)activationParams.GetValueOrDefault("negative_slope") ?? 0.01d;
                act = nn.LeakyReLU(negativeSlope);
                break;

            case "ELU":
                double alpha = (double?)activationParams.GetValueOrDefault("alpha") ?? 1d;
                act = nn.ELU(alpha);
                break;
            
            default:
                throw new ArgumentException($"Activation '{activation}' is not supported.");
        }

        return act;
    }
}