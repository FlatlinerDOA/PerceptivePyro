using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorboard.TensorShapeProto.Types;

namespace PerceptivePyro
{
    public static class TensorExtensions
    {
        public static Tensor normalize(this Tensor input, float p = 2.0f, int dim = 1, float eps = 1e-12f)
        {
            var denom = input.norm(dim: dim, keepdim: true, p: p).clamp_min(eps).expand_as(input);
            return input / denom;
        }
    }
}
