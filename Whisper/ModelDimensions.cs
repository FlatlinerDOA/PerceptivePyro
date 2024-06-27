using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PerceptivePyro.Whisper
{
    public record class ModelDimensions(
        int n_mels,
        int n_audio_ctx,
        int n_audio_state,
        int n_audio_head,
        int n_audio_layer,
        int n_vocab,
        int n_text_ctx,
        int n_text_state,
        int n_text_head,
        int n_text_layer);
}
