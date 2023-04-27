namespace NanoGPTSharp.Encodec;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class SLSTM : nn.Module<Tensor, Tensor>
{
    private bool skip;
    private LSTM lstm;

    //"""
    //LSTM without worrying about the hidden state, nor the layout of the data.
    //Expects input as convolutional layout.
    //"""
    public SLSTM(int dimension, int num_layers = 2, bool skip = true) : base(nameof(SLSTM))
    {
        this.skip = skip;
        this.lstm = nn.LSTM(dimension, dimension, num_layers);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = x.permute(2, 0, 1);
        var (y, _, _) = this.lstm.call(x);
        if (this.skip) {
            y = y + x;
        }
        y = y.permute(1, 2, 0);
        return y;
    }
}
    
