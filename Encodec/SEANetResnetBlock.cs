using static Tensorboard.TensorShapeProto.Types;
using TorchSharp.Modules;
using System.Diagnostics.Contracts;
using Google.Protobuf.WellKnownTypes;

namespace NanoGPTSharp.Encodec;

public sealed class SEANetResnetBlock : nn.Module<Tensor, Tensor>
{
    private nn.Module<Tensor, Tensor> shortcut;
    private Sequential block;

    public SEANetResnetBlock(
        int dim,
        List<int> kernel_sizes = null,
        List<int> dilations = null,
        string activation = "ELU",
        Dictionary<string, object> activation_params = null,
        string norm = "weight_norm",
        Dictionary<string, object> norm_params = null,
        bool causal = false,
        PaddingModes pad_mode = PaddingModes.Reflect,
        int compress = 2,
        bool true_skip = true) : base(nameof(SEANetResnetBlock))
    {
        dilations ??= new() { 1, 1 };
        activation_params ??= new() { { "alpha", 1.0d } };
        kernel_sizes ??= new() { 3, 1 };
        Contract.Assert(kernel_sizes.Count() == dilations.Count(), "Number of kernel sizes should match number of dilations");
        var hidden = dim; // compress
        var blocks = from index_k in kernel_sizes.Zip(dilations, (k, d) => (k, d)).Select((kd, i) => (i, kd.k, kd.d))
                     let i = index_k.i
                     let kernel_size = index_k.k
                     let dilation = index_k.d
                     let in_chs = i == 0 ? dim : hidden
                     let out_chs = i == kernel_sizes.Count - 1 ? dim : hidden
                     select new nn.Module<Tensor, Tensor>[]
                     {
                          activation.get_activation(activation_params),
                          new SConv1d(in_chs, out_chs, kernel_size: kernel_size, dilation: dilation,
                          norm: norm, norm_params: norm_params, causal: causal, pad_mode: pad_mode),
                     };

        this.block = nn.Sequential(blocks.SelectMany(t => t).ToList());


        if (true_skip)
        {
            this.shortcut = nn.Identity();
        } else {
            this.shortcut = new SConv1d(dim, dim, kernel_size: 1, norm: norm, norm_params: norm_params,
                                causal: causal, pad_mode: pad_mode);
        }
    }

    public override Tensor forward(Tensor x)
    {
        return this.shortcut.call(x) + this.block.call(x);
    } 
}
