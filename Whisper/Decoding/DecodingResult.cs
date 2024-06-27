using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.Whisper.Decoding
{
    public record class DecodingResult(
     Tensor audio_features,
     string language,
     Dictionary<string, float>? language_probs = null,
     List<int>? tokens = null,
     string text = "",
     float avg_logprob = float.NaN,
     float no_speech_prob = float.NaN,
     float temperature = float.NaN,
     float compression_ratio = float.NaN);
}
