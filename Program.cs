namespace NanoGPTSharp;

using NanoGPTSharp.Examples;
using System;

internal class Program
{
    const string Help =
        """
        This is just a bunch of examples of using transformer architecture all using pure C# and torch:

        Examples are:

        gpt2_unconditioned - Generates unconditioned random musings by GPT2 - 124M parameter model
        gpt2_large_unconditioned - Generates unconditioned random musings by GPT2 - Large parameters
        gpt2_prompted - Generates a prompted response from GPT2
        gpt3_token_counts - Counts some tokens using GPT3 encoding
        gpt4_token_counts - Counts some tokens using GPT4 encoding
        safetensors - Test code for loading .safetensors files
        training_shakespeare - Training a small language model on Shakespeare. (CUDA GPU with 10gb or more RAM required)

        """;

    static async Task Main(string[] args)
    {
        Task action = args.FirstOrDefault() switch
        {
            "gpt2_unconditioned" => GPTExamples.Gpt2_124M_Unconditioned(),
            "gpt2_large_unconditioned" => GPTExamples.Gpt2_Large_Unconditioned(),
            "gpt2_prompted" => GPTExamples.Gpt2_124m_Prompted(),
            "gpt2_large_prompted" => GPTExamples.Gpt2_Large_Prompted(),
            "gpt3_token_counts" => GPTExamples.Gpt3TokenCounts(),
            "gpt4_token_counts" => GPTExamples.Gpt4TokenCounts(),
            "safetensors" => SafeTensorsExamples.LoadingSafeTensors(),
            "training_shakespeare" => BigramLanguageModelExamples.TrainingOnShakespeare(),
            _ => ShowHelp()
        };

        try
        {
            await action;
        }
        catch(Exception ex) 
        {
            ex.Dump();
        }
    }

    private static Task ShowHelp()
    {
        Help.Dump();
        return Task.CompletedTask;
    }
}