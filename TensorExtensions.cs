using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorboard.TensorShapeProto.Types;

namespace NanoGPTSharp
{
    public static class TensorExtensions
    {
        private static readonly double sqrt_2_pi = Math.Sqrt(2.0 / Math.PI);
        private static readonly double sqrt_2 = Math.Sqrt(2.0);

        public static Tensor Gelu(this Tensor x) => x * 0.5 * (1.0 + torch.erf(x / sqrt_2));
        
        /// <summary>
        /// Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        /// Reference: <a href="https://arxiv.org/abs/1606.08415">Gaussian Error Linear Units(GELU) paper</a>
        /// </summary>
        /// <param name="x">Input tensor</param>
        /// <returns>Tensor of gelu operator applied.</returns>
        public static Tensor NewGelu(this Tensor x) => 0.5 * x * (1.0 + torch.tanh(sqrt_2_pi * (x + 0.044715 * torch.pow(x, 3.0))));
        
        public static Tensor FastGELU(this Tensor x) => 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)));
        
        public static Tensor QuickGELU(this Tensor x) => x * torch.sigmoid(1.702 * x);
        
        public static Tensor ClippedGELU(this Tensor x, float min , float max) => torch.clip(x.Gelu(), min, max);
        
        
        public static nn.Module<Tensor, Tensor> GetActivationFunction(this string activation_func) => activation_func switch
        {
            "gelu" => nn.GELU(),
            "gelu_10" => new Activation(t => t.ClippedGELU(-10, 10), nameof(ClippedGELU)),
            "gelu_fast" => new Activation(FastGELU, nameof(FastGELU)),
            "gelu_new" => new Activation(NewGelu, nameof(NewGelu)),
            "gelu_python" =>  new Activation(Gelu, nameof(Gelu)),
            // "gelu_pytorch_tanh" => PytorchGELUTanh,
            // "gelu_accurate" => AccurateGELUActivation,
            // "laplace" => LaplaceActivation,
            // "linear" => LinearActivation,
            // "mish" => MishActivation,
            "quick_gelu" => new Activation(QuickGELU, nameof(QuickGELU)),
            "relu" => nn.ReLU(),
            ///"relu2" => ReLUSquaredActivation,
            "relu6" => nn.ReLU6(),
            "sigmoid" => nn.Sigmoid(),
            "silu" => nn.SiLU(),
            "swish" => nn.SiLU(),
            "tanh" => nn.Tanh(),
            _ => throw new NotSupportedException($"{activation_func} not supported.")
        };
        
        public static Tensor apply_chunking_to_forward(this  Tensor input_tensors, Func<Tensor, Tensor> forward_fn, int chunk_size, int chunk_dim)
        {
            Func<Tensor[], Tensor> forward_array = (t) => forward_fn(t[0]);
            return apply_chunking_to_forward(forward_array, chunk_size, chunk_dim, input_tensors);
        }

        /// <summary>
        /// This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
        /// `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
        /// If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
        /// applying `forward_fn` to `input_tensors`.
        /// </summary>
        /// <param name="foward_fn"></param>
        /// <param name="chunk_size"></param>
        /// <param name="chunk_dim"></param>
        /// <param name="input_tensors"></param>
        /// <typeparam name="TResult"></typeparam>
        /// <returns></returns>
        public static Tensor apply_chunking_to_forward(this Func<Tensor[], Tensor> forward_fn, int chunk_size, int chunk_dim, params Tensor[] input_tensors)
        {
            Contract.Assert(input_tensors.Length > 0, $"{input_tensors} has to be a list of tensors");
            if (chunk_size > 0)
            {
                var tensor_shape = input_tensors[0].shape[chunk_dim];
                foreach (var input_tensor in input_tensors)
                {
                    if (input_tensor.shape[chunk_dim] != tensor_shape)
                    {
                        throw new InvalidOperationException($"All input tenors have to be of the same shape: {tensor_shape}, found shape {input_tensor.shape[chunk_dim]}");
                    }
                }

                Contract.Assert(input_tensors[0].shape[chunk_dim] % chunk_size == 0, $"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk  size {chunk_size}");

                var num_chunks = input_tensors[0].shape[chunk_dim]; // chunk_size

                // chunk input tensor into tuples
                var input_tensors_chunks = from input_tensor in input_tensors
                                                            select input_tensor.chunk(num_chunks, dim:chunk_dim);
                // apply forward fn to every tuple
                var output_chunks = (from input_tensors_chunk in input_tensors_chunks
                                               select forward_fn(input_tensors_chunk)).ToList();
                // concatenate output at same dimension
                return torch.cat(output_chunks, dim: chunk_dim);
            }
    
            return forward_fn(input_tensors);
        }
        
        public static Tensor normalize(this Tensor input, float p = 2.0f, int dim = 1, float eps = 1e-12f)
        {
            var denom = input.norm(dim: dim, keepdim: true, p: p).clamp_min(eps).expand_as(input);
            return input / denom;
        }
        
        public static IReadOnlyList<T> Slice<T>(this IReadOnlyList<T> list, Range range)
        {
            var (offset, length) = range.GetOffsetAndLength(list.Count);
            return list.Skip(offset).Take(length).ToList();
        }
        
        public static IReadOnlyList<T> Slice<T>(this IReadOnlyList<T> list, Index index)
        {
            var offset = index.GetOffset(list.Count);
            return list.Skip(offset).Take(1).ToList();
        }
        
        /// <summary>
        /// Wraps a function into a Pytorch module so that the model is self describing
        /// as to what activation function is used, when printing out the model.
        /// </summary>
        private class Activation : nn.Module<Tensor, Tensor>
        {
            private readonly Func<Tensor,Tensor> activation;

            public Activation(Func<Tensor, Tensor> activation, string name): base(name)
            {
                this.activation = activation;
            }

            public override Tensor forward(Tensor input) => this.activation(input);
        }
    }
}
