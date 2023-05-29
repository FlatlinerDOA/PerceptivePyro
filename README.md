# Perceptive Pyro

This library allows you to run selected Transformer based Large Language Models using just .NET and TorchSharp.

### Key benefits
* No python dependency
* Just Nuget Package references
* Can self provision models at runtime

### Use it to
* Train GPT-2 based models on your own data.
* Perform classification tasks
* Generate vector embeddings for storing in a vector database such as Pinecone, ChromaDb or Milvus
* Perform semantic similarity scoring use a Roberta based model (all-distilroberta-v1).

### History
This is started as a learning exercise following along with [Andre Karpathy's Lets Build GPT: From scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3510s).
GPT model from the ground up in C# using [TorchSharp](https://github.com/dotnet/TorchSharp)

### Using Examples:

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
