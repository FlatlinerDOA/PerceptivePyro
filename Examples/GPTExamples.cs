﻿namespace NanoGPTSharp.Examples;

using SharpToken;

internal class GPTExamples
{
    public static async Task Gpt2_124M_Unconditioned()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT2 pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        // tiktoken style encoding of text into BPE (Byte pair encodings)
        var encoding = GptEncoding.GetEncoding("r50k_base");
        
        // Start with a single empty token as the starting context, so that GPT2 can get creative with what comes next.
        var gpt_context = torch.zeros(new[] { 1L, 1L }, dtype: torch.@long, device: device);
        
        // Run the prediction for up to 200 tokens.
        var raw_output = gpt.generate(gpt_context, max_new_tokens: 200);
        
        // Decode back into human readable form
        encoding.Decode(raw_output[0].data<long>().Select(v => (int)v).ToList()).Dump();
    }

    public static async Task Gpt2_Large_Unconditioned()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT2 pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2-large", device);

        // tiktoken style encoding of text into BPE (Byte pair encodings)
        var encoding = GptEncoding.GetEncoding("r50k_base");

        // Start with a single empty token as the starting context, so that GPT2 can get creative with what comes next.
        var gpt_context = torch.zeros(new[] { 1L, 1L }, dtype: torch.@long, device: device);
        $"Context Shape: {gpt_context.shape.Stringify()}".Dump();

        // Run the prediction for up to 200 tokens.
        var raw_output = gpt.generate(gpt_context, max_new_tokens: 200);

        // Decode back into human readable form
        encoding.Decode(raw_output[0].data<long>().Select(v => (int)v).ToList()).Dump();
    }

    public static async Task Gpt2_124m_Prompted()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2", device);

        // Some text we are giving GPT2 to riff on.
        generator(gpt, "Hello, I'm a language model,").ToList().Dump();
    }

    public static async Task Gpt2_Large_Prompted()
    {
        // Fix the randomness in place.
        set_seed(1337);

        // Load GPT pre-trained weights.
        const string device = "cpu";
        var gpt = await GPTModel.from_pretrained("gpt2-large", device);

        // Some text we are giving GPT2 to riff on.
        generator(gpt, "Hello, I'm a language model,").ToList().Dump();
    }

    public static async Task Gpt3TokenCounts()
    {
        // https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        // p50k_base: Codex models, text-davinci-002, text-davinci-003
        var encoding = GptEncoding.GetEncoding("cl100k_base");

        var prompt = "The quick brown fox jumps over the lazy dog";
        var tokenCount = encoding.Encode(prompt).Count;
        $"{tokenCount} tokens are in {prompt}".Dump();
    }

    public static async Task Gpt4TokenCounts()
    {
        // https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        // cl100k_base:	gpt-4, gpt-3.5-turbo, text-embedding-ada-002
        var encoding = GptEncoding.GetEncoding("cl100k_base");

        var prompt = "The quick brown fox jumps over the lazy dog";
        var tokenCount = encoding.Encode(prompt).Count;
        $"{tokenCount} tokens are in {prompt}".Dump();
    }

    public static async Task FineTuning()
    {
        // TODO: We're gonna need a bigger GPU...
        var gpt = await GPTModel.from_pretrained("gpt2", "cpu");
        var encoding = GptEncoding.GetEncoding("r50k_base");
        gpt.train();
    }

    private static IEnumerable<string> generator(GPTModel gpt, string prompt, int max_length = 30, int num_return_sequences = 1, string device = "cpu")
    {
        var encoding = GptEncoding.GetEncoding("r50k_base");
        var encoded_prompt = encoding.Encode(prompt, new HashSet<string>() { "<|endoftext|>" });
        var gpt_context = torch.as_tensor(encoded_prompt, dtype: @long, device: device).reshape(1, encoded_prompt.Count);
        gpt_context.Dump();
        for (int i = 0; i < num_return_sequences; i++)
        {
            var output = gpt.generate(gpt_context, max_new_tokens: max_length, temperature: 0.8d, top_k: 200);
            output.Dump();
            yield return encoding.Decode(output[0].data<long>().Select(v => (int)v).ToList());      
        }
    }

    private static void set_seed(int seed)
    {
        torch.manual_seed(seed);
        torch.cuda.manual_seed(seed);
        torch.random.manual_seed(seed);
        torch.backends.cuda.matmul.allow_tf32 = true; // allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = true; // allow tf32 on cudnn
    }
}