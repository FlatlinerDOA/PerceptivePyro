namespace PerceptivePyro.Whisper.Decoding
{
    /// <summary>
    /// Select the sample with the highest log probabilities, penalized using either 
    /// a simple length normalization or Google NMT paper's length penalty
    /// </summary>
    public class MaximumLikelihoodRanker : SequenceRanker
    {
        private readonly float? length_penalty;

        public MaximumLikelihoodRanker(float? length_penalty)
        {
            this.length_penalty = length_penalty;
        }

        public override List<int> Rank(List<List<Tensor>> tokens, List<List<float>> sum_logprobs)
        {
            List<float> scores(List<float> logprobs, List<int> lengths)
            {
                var result = new List<float>();
                foreach (var (logprob, length) in logprobs.Zip(lengths))
                {
                    // from the Google NMT paper
                    float penalty = this.length_penalty is null ? length : (float)Math.Pow(((5 + length) / 6), this.length_penalty.Value);
                    result.Add(logprob / penalty);
                }

                return result;
            }

            // get the sequence with the highest score
            var lengths = tokens.Select(s => s.Select(t => t.AsEnumerable().Count()).ToList()).ToList();
            return sum_logprobs.Zip(lengths)
                .Select(x => torch.argmax(torch.tensor(scores(x.First, x.Second))).item<int>())
                .ToList();
        }
    }
}
