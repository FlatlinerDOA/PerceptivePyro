# NanoGPTSharp

This is a learning exercise following along with [Andre Karpathy's Lets Build GPT: From scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3510s).

We are building a small GPT model from the ground up in C# using [TorchSharp](https://github.com/dotnet/TorchSharp)

Usage:

```
This is just a bunch of examples of using transformer architecture all using pure C# and torch:

nanogptsharp {command}

{command}:
* gpt2_unconditioned - Generates unconditioned random musings by GPT2
* gpt2_prompted - Generates a prompted response from GPT2
* gpt3_token_counts - Counts some tokens using GPT3 encoding
* gpt4_token_counts - Counts some tokens using GPT4 encoding
* safetensors - Test code for loading .safetensors files
* training_shakespeare - Training a small language model on Shakespeare. (CUDA GPU with 10gb or more RAM required)

```



NOTE: The code has the following global usings in all files:

```
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
```

