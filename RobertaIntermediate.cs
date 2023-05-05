namespace NanoGPTSharp;
public class RobertaIntermediate : nn.Module<Tensor, Tensor>
{
    private readonly Linear dense;
    private readonly nn.Module<Tensor, Tensor> intermediate_act_fn;

    public RobertaIntermediate(RobertaConfig config) : base(nameof(RobertaIntermediate))
    {
        this.dense = nn.Linear(config.hidden_size, config.intermediate_size);
        this.intermediate_act_fn = config.hidden_act is nn.Module<Tensor, Tensor> m ? m : (config.hidden_act?.ToString() ?? "gelu").GetActivationFunction();
    }

    public override Tensor forward(Tensor hidden_states)
    {
        hidden_states = this.dense.call(hidden_states);
        hidden_states = this.intermediate_act_fn.call(hidden_states);
        return hidden_states;
    }
}
