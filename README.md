# PerceptivePyro

This is a learning exercise following along with [Andre Karpathy's Lets Build GPT: From scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3510s).

We are building a small GPT model from the ground up in C# using [TorchSharp](https://github.com/dotnet/TorchSharp)

## Capabilities
* GPT-2 training and/or inferance
* Sentence Embedding for Question Answering

## Usage

```
This is just a bunch of examples of using transformer architecture all using pure C# and torch:

PerceptivePyro {command}

{command}:
* benchmark_msmarco - Evaluates GPT2 embedding sentence similarity scoring on the MS MARCO V2.1 dataset.
* benchmark_sick - Evaluates GPT2 embedding sentence similarity scoring on the SICK dataset.
* gpt2_unconditioned - Generates unconditioned random musings by GPT2 - 124M parameter model
* gpt2_large_embeddings - Generates embeddings for a sentance - 
* gpt2_large_unconditioned - Generates unconditioned random musings by GPT2 - Large parameters
* gpt2_prompted - Generates a prompted response from GPT2
* gpt3_token_counts - Counts some tokens using GPT3 encoding
* gpt4_token_counts - Counts some tokens using GPT4 encoding
* roberta_similarity - Compares sentence similarity using the all-distilroberta-v1 model.
* safetensors - Test code for loading .safetensors files
* training_shakespeare - Training a small language model on Shakespeare (CUDA GPU with 10gb or more RAM required).

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

