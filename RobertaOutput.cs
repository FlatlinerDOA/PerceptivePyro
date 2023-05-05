namespace NanoGPTSharp;
public class RobertaOutput : nn.Module<Tensor, Tensor, Tensor>
{
    private readonly RobertaConfig config;
    private readonly Linear dense;
    private readonly TorchSharp.Modules.LayerNorm LayerNorm;
    private readonly Dropout dropout;

    public RobertaOutput(RobertaConfig config) : base(nameof(RobertaOutput))
    {
        this.config = config;
        this.dense = nn.Linear(config.intermediate_size, config.hidden_size);
        this.LayerNorm = nn.LayerNorm(config.hidden_size, eps:config.layer_norm_eps);
        this.dropout = nn.Dropout(config.hidden_dropout_prob);
        this.RegisterComponents();
    }

    public override Tensor forward(Tensor hidden_states, Tensor input_tensor)
    {
        hidden_states = this.dense.call(hidden_states);
        hidden_states = this.dropout.call(hidden_states);
        hidden_states = this.LayerNorm.call(hidden_states + input_tensor);
        return hidden_states;
    }
}
