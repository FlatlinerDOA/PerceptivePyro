
namespace NanoGPTSharp.Examples;

public static class SafeTensorsExamples
{
    /// <summary>
    /// Demonstrates loading and inspecting safe tensors.
    /// </summary>
    /// <returns></returns>
    public static async Task Loading_Safe_Tensors()
    {
        await GPTModel.from_pretrained("gpt2", "cpu"); // Ensure we have it downloaded first.
        var list = SafeTensors.LoadFile(@".\models\gpt2\model.safetensors", "cpu").ToList();
        list.Dump();
    }
}
