namespace NanoGPTSharp;

public class RobertaPooler : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    private readonly Tanh activation;

    public RobertaPooler(RobertaConfig config) : base(nameof(RobertaPooler))
    {
        this.dense = nn.Linear(config.hidden_size, config.hidden_size);
        this.activation = nn.Tanh();
    }

    public override Tensor forward(Tensor hidden_states)
    {
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token.
        var first_token_tensor = hidden_states[.., 0];
        var pooled_output = this.dense.call(first_token_tensor);
        pooled_output = this.activation.call(pooled_output);
        return pooled_output;
    }
}

