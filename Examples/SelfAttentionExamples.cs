using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using F = TorchSharp.torch.nn.functional;

namespace PerceptivePyro.Examples
{
    internal class SelfAttentionExamples
    {
        /// <summary>
        /// Demonstrates how self attention is constructed, step by step.
        /// </summary>
        /// <returns></returns>
        internal static Task self_attention_explained()
        {
            // Self attention - we are attending to our ourselves (in this case the x input).
            // Cross attention - we are attending to data from elsewhere.
            var (B, T, C) = (4, 8, 32); // batch, time, channels
            var x = torch.randn(B, T, C); // some random test input data.

            const int head_size = 16;
            var key = torch.nn.Linear(C, head_size, hasBias: false); // Content of the character in question.
            var query = torch.nn.Linear(C, head_size, hasBias: false); // What we are looking for.
            var value = torch.nn.Linear(C, head_size, hasBias: false); // What do I bring to the table?
            var k = key.call(x);
            var q = query.call(x);

            var wei = q.matmul(k.transpose(-2, -1)); // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
                                                     //var wei = torch.zeros(new long[] { T, T }); // start wei with 2d matrix of zeroes (T, T).

            var tril = torch.tril(torch.ones(T, T)); // triangular matrix of 1's for time so that the past see the future.
            wei = wei.masked_fill(tril == 0, float.NegativeInfinity); // mask fill wei, with the zeroes from tril replaced with -Inf. This gives wei 0's and -Inf in a triangle.
            wei = F.softmax(wei, dim: -1); // Softmax replaces -Inf with 0 and weights the zeroes evenly distributed by row.

            wei.Dump();

            var v = value.call(x);
            var output = wei.matmul(v);
            output.shape.Dump();
            output.Dump();

            return Task.CompletedTask;
        }
    }
}
