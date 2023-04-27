namespace NanoGPTSharp.Encodec;

using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SEANetEncoder : Module<Tensor, Tensor>
{
    private int channels;
    private int dimension;
    private int n_filters;
    private List<int> ratios;
    private int n_residual_layers;
    private int hop_length;
    private Sequential model;

    public SEANetEncoder(int channels = 1, int dimension = 128, int n_filters = 32, int n_residual_layers = 1,
                         List<int> ratios = null, string activation = "ELU", Dictionary<string, object> activation_params = null,
                         string norm = "weight_norm", Dictionary<string, object> norm_params = null, int kernel_size = 7,
                         int last_kernel_size = 7, int residual_kernel_size = 3, int dilation_base = 2, bool causal = false,
                         string pad_mode = "reflect", bool true_skip = false, int compress = 2, int lstm = 2)
        : base("SEANetEncoder")
    {
        this.channels = channels;
        this.dimension = dimension;
        this.n_filters = n_filters;
        this.ratios = new List<int>(ratios);
        this.ratios.Reverse();
        this.n_residual_layers = n_residual_layers;
        this.hop_length = 1;
        foreach (var ratio in this.ratios)
        {
            this.hop_length *= ratio;
        }

        // The activation function
        var act = get_activation(activation, activation_params);

        int mult = 1;
        var modelLayers = new List<Module>();

        // Add the first convolution layer
        modelLayers.Add(new SConv1d(channels, mult * n_filters, kernel_size, norm, norm_params, causal, pad_mode));

        // Downsample to raw audio scale
        for (int i = 0; i < this.ratios.Count; i++)
        {
            int ratio = this.ratios[i];
            // Add residual layers
            for (int j = 0; j < n_residual_layers; j++)
            {
                modelLayers.Add(new SEANetResnetBlock(mult * n_filters, new List<int> { residual_kernel_size, 1 },
                                                      new List<int> { (int)Math.Pow(dilation_base, j), 1 },
                                                      norm, norm_params, activation, activation_params,
                                                      causal, pad_mode, compress, true_skip));
            }

            // Add downsampling layers
            modelLayers.Add(act);
            modelLayers.Add(new SConv1d(mult * n_filters, mult * n_filters * 2, ratio * 2, ratio, norm, norm_params, causal, pad_mode));

            mult *= 2;
        }

        if (lstm > 0)
        {
            modelLayers.Add(new SLSTM(mult * n_filters, lstm));
        }

        modelLayers.Add(act);
        modelLayers.Add(new SConv1d(mult * n_filters, dimension, last_kernel_size, norm, norm_params, causal, pad_mode));

        this.model = Sequential(modelLayers.ToArray());
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        return this.model.forward(x);
    }

    private Module get_activation(string activation, Dictionary<string, object> activation_params)
    {
        switch (activation)
        {
            case "ELU":
                return new ELU((double)activation_params["alpha"]);
            // Add more activation functions as needed.
            default:
                throw new ArgumentException($"Unsupported activation function");
        }
    }
}
    
