namespace NanoGPTSharp.Encodec;

using System;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Text;
using System.Threading.Channels;
using System.Xml.Linq;
using TorchSharp.Modules;
using static NanoGPTSharp.Encodec.EncodecModel;
using static System.Reflection.Metadata.BlobBuilder;
using static Tensorboard.CostGraphDef.Types;
using F = TorchSharp.torch.nn.functional;

internal class EncodecModel : nn.Module<Tensor, Tensor>
{
    private float? bandwidth;
    private List<float> target_bandwidths;
    private SEANetEncoder encoder;
    private ResidualVectorQuantizer quantizer;
    private SEANetDecoder decoder;
    private float? segment;
    private float overlap;
    private object frame_rate;
    private int sample_rate;
    private int channels;
    private bool normalize;

    private int? segment_length => this.segment is null ? null : (int)(this.overlap * this.sample_rate);

    private int bits_per_codebook;
    private int segment_stride;

    public EncodecModel(
        SEANetEncoder encoder,
        SEANetDecoder decoder ,
        ResidualVectorQuantizer quantizer,
        List<float> target_bandwidths,
        int sample_rate,
        int channels,
        bool normalize = false,
        float? segment = null,
        float overlap = 0.01f,
        string name = "unset")
    {
        this.bandwidth = null;
        this.target_bandwidths = target_bandwidths;
        this.encoder = encoder;
        this.quantizer = quantizer;
        this.decoder = decoder;
        this.sample_rate = sample_rate;
        this.channels = channels;
        this.normalize = normalize;
        this.segment = overlap;
        this.overlap = overlap;
        this.frame_rate = Math.Ceiling(this.sample_rate / this.encoder.ratios.Sum());
        this.name = name;
        this.bits_per_codebook = (int)Math.Log2(this.quantizer.bins);
        Contract.Assert(2 * this.bits_per_codebook == this.quantizer.bins, "quantizer bins must be a power of 2");
    }


    /// <summary>
    /// Given a tensor `x`, returns a list of frames containing
    /// the discrete encoded codes for `x`, along with rescaling factors
    /// for each segment, when `self.normalize` is True.
    /// Each frames is a tuple `(codebook, scale)`, with `codebook` of
    /// shape `[B, K, T]`, with `K` the number of codebooks.
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    internal List<EncodedFrame> encode(Tensor x)
    {
        /* def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames
        */
        Contract.Assert(x.dim() == 3, "Expected 3 dimensional tensor.");
        var (_, channels, length) = (x.shape[0], x.shape[1], (int)x.shape[2]);
        Contract.Assert(channels > 0 && channels <= 2, "Expected 1 or 2 channels");

        var segment_length = (int)this.segment_length;
        var stride = 0;
        if (segment_length == 0)
        {
            segment_length = length;
            stride = length;
        }
        else
        {
            stride = this.segment_stride;  // type: ignore
        }

        Contract.Assert(stride != 0, "stride cannot be zero.");
        var encoded_frames = new List<EncodedFrame>(length);
        for (int offset = 0; offset < length; offset += stride)
        {
            var frame = x[.., .., offset..(offset + segment_length)];
            encoded_frames.Add(this._encode_frame(frame));
        }
        return encoded_frames;

    }

    private EncodedFrame _encode_frame(Tensor frame)
    {
        throw new NotImplementedException();
    }

    public override Tensor forward(Tensor input)
    {
        var frames = this.encode(input);
        return this.decode(frames)[.., .., ..x.shape[-1]];
    }

    private Tensor decode(List<EncodedFrame> frames)
    {
        throw new NotImplementedException();
    }

    internal record EncodedFrame(Tensor tensor, Tensor? other);
}
