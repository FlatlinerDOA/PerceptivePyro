# Perceptive Pyro

This library allows you to run selected Transformer based Large Language Models using just .NET and TorchSharp.

### Key benefits
* No python dependency
* Nuget package references to TorchSharp and SharpToken only
* Can self provision pre-trained models from HuggingFace at runtime (uses .safetensors).

### Use it to
* Load GPT-2 pre-trained weights from HuggingFace and perform inference using it.
* Train your own GPT-2 style models on your own data.
* Load a Roberta pre-trained model (all-distilroberta-v1) to generate vector embeddings for storing in a vector database such as Pinecone, ChromaDb or Milvus
* Use a Roberta pre-trained model (all-distilroberta-v1) to perform semantic similarity scoring of sentences.

### History
This started as a learning exercise following along with [Andre Karpathy's Lets Build GPT: From scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3510s).
Let's train a GPT model from the ground up in C# using [TorchSharp](https://github.com/dotnet/TorchSharp)

### Examples:

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
