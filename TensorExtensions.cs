using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro;

public static class TensorExtensions
{
    public static Tensor normalize(this Tensor input, float p = 2.0f, int dim = 1, float eps = 1e-12f)
    {
        var denom = input.norm(dim: dim, keepdim: true, p: p).clamp_min(eps).expand_as(input);
        return input / denom;
    }

    /// <summary>
    /// Useful for calculating embeddings, by taking token embeddings and calculating their mean values
    /// Shape input is (Batch, Time, Embedding Channel) to output (Batch, Embedding Channel).
    /// </summary>
    /// <param name="model_output"></param>
    /// <param name="attention_mask"></param>
    /// <returns>Tensor of shape (Batch, Embedding Channel)</returns>
    public static Tensor mean_pooling(this Tensor model_output, Tensor attention_mask)
    {
        var token_embeddings = model_output; // token embeddings
        var input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).@float();
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min: 1e-9);
    }
}
