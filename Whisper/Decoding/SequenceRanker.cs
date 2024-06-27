using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.Whisper.Decoding
{
    public abstract class SequenceRanker
    {
        public abstract List<int> Rank(List<List<Tensor>> tokens, List<List<float>> sum_logprobs);
    }
}
