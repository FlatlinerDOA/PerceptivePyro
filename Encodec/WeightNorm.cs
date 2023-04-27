namespace NanoGPTSharp.Encodec;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;

internal static partial class ConvExtensions
{
    public class WeightNorm
    {
        private string _name;
        private int _dim;

        public WeightNorm(string name, int? dim = null)
        {
            _dim = dim ?? -1;
            _name = name;
        }

        public static Module<Tensor, Tensor> Apply(Module<Tensor, Tensor> module, string name, int? dim = null)
        {
            // Check for existing WeightNorm hooks
            //if (module.ForwardPreHookKeys.Any(k => module.GetForwardPreHook(k) is WeightNorm hook && hook.this._name == name))
            //{
            //    throw new InvalidOperationException($"Cannot register two weight_norm hooks on the same parameter {name}");
            //}

            if (dim == null)
            {
                dim = -1;
            }

            var fn = new WeightNorm(name, dim.Value);

            var weight = module.get_parameter(name);

            // Remove 'weight' from the parameter list
            module.register_parameter(name, null);

            // Add 'g' and 'v' as new parameters and express 'weight' as g/||v|| * v
            module.register_parameter(name + "_g", new Parameter(NormExceptDim(weight, 2, dim.Value)));
            module.register_parameter(name + "_v", new Parameter(weight));
            module.get_parameter(name).set_(fn.ComputeWeight(module));

            // Recompute weight before every forward()
            module.register_forward_pre_hook((module, input) => fn.ComputeWeight(module));
            return module;
        }

        // TODO: Make return type more specific
        public Tensor ComputeWeight(Module<Tensor, Tensor> module)
        {
            var g = module.get_parameter(this._name + "_g");
            Tensor v = module.get_parameter(this._name + "_v");
            return WeightNormFunction(v!, g!, _dim);
        }

        public void remove(Module<Tensor, Tensor> module)
        {
            var weight = ComputeWeight(module);
            module.register_parameter(this._name, null);
            module.register_parameter(this._name + "_g", null);
            module.register_parameter(this._name + "_v", null);
            module.register_parameter(_name, new Parameter(weight));
        }

        public void call(Module<Tensor, Tensor> module)
        {
            module.get_parameter(this._name).set_(ComputeWeight(module));
        }

        // Implement the weight normalization function here
        private static Tensor WeightNormFunction(Tensor v, Tensor g, int dim)
        {
            // TODO: Implement the weight normalization function
            throw new NotImplementedException();
        }

        private static Tensor NormExceptDim(Tensor input, float p, int dim)
        {
            var dims_to_reduce = Enumerable.Range(0, (int)input.dim()).Select(i => i == dim ? 0..0 : (TensorIndex)(0..(int)input.shape[i])).ToArray();
            return input[dims_to_reduce].norm(p);
        }
    }
}
