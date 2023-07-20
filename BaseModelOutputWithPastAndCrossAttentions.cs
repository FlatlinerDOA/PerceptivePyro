namespace PerceptivePyro;

using System.Collections.Generic;

public record BaseModelOutputWithPastAndCrossAttentions(
    Tensor last_hidden_state,
    Tensor? pooled_output,
    IReadOnlyList<Tensor> past_key_values,
    IReadOnlyList<Tensor> hidden_states,
    IReadOnlyList<Tensor> attentions,
    IReadOnlyList<Tensor> cross_attentions);
