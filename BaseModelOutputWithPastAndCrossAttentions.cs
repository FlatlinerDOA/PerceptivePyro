namespace NanoGPTSharp;

using System.Collections.Generic;

public record BaseModelOutputWithPastAndCrossAttentions(
    Tensor last_hidden_state,
    IReadOnlyList<Tensor> past_key_values,
    IReadOnlyList<Tensor> hidden_states,
    IReadOnlyList<Tensor> attentions,
    IReadOnlyList<Tensor> cross_attentions);
