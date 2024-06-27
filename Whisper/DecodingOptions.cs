using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PerceptivePyro.Whisper.Decoding;

namespace PerceptivePyro.Whisper
{
    public record class DecodingOptions(
      string task = "transcribe", //  whether to perform X->X "transcribe" or X->English "translate"
      string? language = null, // language that the audio is in; uses detected language if None
      float temperature = 0.0f, // sampling-related options
      int? sample_len = null, // maximum number of tokens to sample
      int? best_of = null, // number of independent sample trajectories, if t > 0
      int? beam_size = null, // number of beams in beam search, if t == 0
      float? patience = null, // patience in beam search (arxiv:2204.05424)
      float? length_penalty = null, // "alpha" in Google NMT, or None for length norm, when ranking generations to select which to return among the beams or best-of-N samples
      Prompt? prompt = null, // text or tokens to feed as the prompt or the prefix; for more info: https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
      Prompt? prefix = null, // for the previous context to prefix the current context
      string suppress_tokens = "-1", // list of tokens ids (or comma-separated token ids) to suppress "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
      bool suppress_blank = true, // this will suppress blank outputs
      bool without_timestamps = false, // timestamp sampling options use <|notimestamps|> to sample text tokens only 
      float? max_initial_timestamp = 1.0f,
      bool fp16 = true // implementation details: use fp16 for most of the calculation
      )
    {
        public DecodingOptions With(Dictionary<string, object>? kwargs = null)
        {
            // TODO: Replace keys
            return this;
        }
    }
}
