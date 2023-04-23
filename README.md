# NanoGPTSharp

This is a learning exercise following along with [Andre Karpathy's Lets Build GPT: From scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3510s).
We are building a small GPT model from the ground up in C# using [TorchSharp](https://github.com/dotnet/TorchSharp)

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