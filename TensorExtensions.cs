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
    }
}
