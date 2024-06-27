# Whisper model

Speech-to-text recognition model from OpenAI (MIT License).
Also supports language detection and low level MEL spectrogram audio token decoding.
Code ported to C# from [Whisper](https://github.com/openai/whisper) repository on Github.

Uses only TorchSharp, all numpy usage has been removed to simplify dependencies.

## C# Usage:

```csharp
using PerceptivePyro.Whisper;

var model = WhisperModel.load_model("base");
var result = model.transcribe("audio.mp3");
Console.WriteLine(result["text"]);
```


```csharp
using PerceptivePyro.Whisper;

var model = WhisperModel.load_model("base");

// load audio and pad/trim it to fit 30 seconds
var audio = whisper.load_audio("audio.mp3");
audio = WhisperModel.pad_or_trim(audio);

// make log-Mel spectrogram and move to the same device as the model
var mel = WhisperModel.log_mel_spectrogram(audio).to(model.device);

// detect the spoken language
var (_, probs) = model.detect_language(mel);
Console.WriteLine($"Detected language: {max(probs, key=probs.get)}");

// decode the audio
var options = new DecodingOptions();
var result = whisper.decode(model, mel, options)

# print the recognized text
Console.WriteLine(result.text);

```