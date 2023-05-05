using System.Linq.Expressions;
using System.Reflection;
using System.Xml.XPath;
using NanoGPTSharp.Examples;

namespace NanoGPTSharp;

internal class Program
{
    private static readonly Type[] ExampleTypes = new[]
    {
        typeof(GPTExamples),
        typeof(BigramLanguageModelExamples),
        typeof(RobertaExamples),
        typeof(SafeTensorsExamples),
        typeof(SelfAttentionExamples)
    };

    const string Help =
        """
        This is just a bunch of examples of using transformer architecture all using pure C# and torch:

        Examples are:

        benchmark_msmarco - Evaluates GPT2 embedding sentence similarity scoring on the MS MARCO V2.1 dataset.
        benchmark_sick - Evaluates GPT2 embedding sentence similarity scoring on the SICK dataset.
        gpt2_unconditioned - Generates unconditioned random musings by GPT2 - 124M parameter model
        gpt2_large_embeddings - Generates embeddings for a sentance - 
        gpt2_large_unconditioned - Generates unconditioned random musings by GPT2 - Large parameters
        gpt2_prompted - Generates a prompted response from GPT2
        gpt3_token_counts - Counts some tokens using GPT3 encoding
        gpt4_token_counts - Counts some tokens using GPT4 encoding
        roberta_similarity - Compares sentence similarity using the all-distilroberta-v1 model.
        safetensors - Test code for loading .safetensors files
        training_shakespeare - Training a small language model on Shakespeare. (CUDA GPU with 10gb or more RAM required)

        """;

    static async Task Main(string[] args)
    {
        try
        {
            var examples = GetExamples(ExampleTypes).ToDictionary(k => k.Name, k => k.Function, StringComparer.OrdinalIgnoreCase);
            await examples.GetValueOrDefault(args.FirstOrDefault(), ShowHelp)();
        }
        catch (Exception ex)
        {
            ex.Dump();
        }
    }

    private static IEnumerable<(string Name, Func<Task> Function, string Help)> GetExamples(IEnumerable<Type> exampleTypes)
    {
        var q = from type in exampleTypes
                from method in type.GetMethods(BindingFlags.Static | BindingFlags.Public)
                select (
                    Name: method.Name.ToLowerInvariant(),
                    Function: Expression.Lambda<Func<Task>>(Expression.Call(null, method)).Compile(),
                    Help: GetHelpForMethod(type, method)
                );
        return q;
    }

    private static string GetHelpForMethod(Type type, MethodInfo method)
    {
        var xmlFilename = $"{type.Assembly.GetName().Name}.xml";
        var doc = new XPathDocument(xmlFilename);
        doc.Dump();
        return null;
    }

    private static Task ShowHelp()
    {
        Help.Dump();
        return Task.CompletedTask;
    }
}