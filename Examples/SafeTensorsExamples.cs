
namespace NanoGPTSharp.Examples;

public static class SafeTensorsExamples
{
    public static async Task LoadingSafeTensors()
    {
        var list = SafeTensors.LoadFile(@".\models\gpt2\model.safetensors", "cpu").ToList();
        list.Dump();
    }
}
