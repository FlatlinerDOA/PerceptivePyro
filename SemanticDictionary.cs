namespace PerceptivePyro;

/// <summary>
/// A semantic dictionary allows searching for nearest neighbours by the semantic content of the <typeparamref name="TValue"/>.
/// You provide a content selector function to get the semantic content to be indexed. An embedding is created for that content and then you can perform <see cref="GetBatchTop"/> to get the top N most similar results. 
/// </summary>
/// <typeparam name="TKey">The key of the content being indexed.</typeparam>
/// <typeparam name="TValue">The content of the content being indexed.</typeparam>
public sealed class SemanticDictionary<TKey, TValue> : IDictionary<TKey, TValue>
{
    private static readonly SearchResultScoreComparer ScoreComparer = new();
    
    private static readonly CorpusKeyRangeComparer CorpusRangeComparer = new();

    private readonly Func<TValue, string> contentSelector;
    private readonly ReaderWriterLockSlim readWrite = new ReaderWriterLockSlim(LockRecursionPolicy.NoRecursion);
    private readonly Func<Tensor, Tensor, Tensor> scoreFunction;
    private readonly Dictionary<TKey, (Range CorpusRange, TValue Value)> values = new();
    private readonly List<(int StartIndex, int EndIndex, TKey Key, List<IReadOnlyList<int>> Excerpts)> corpusKeyRanges = new();
    private Tensor corpusEmbeddings;

    private SemanticDictionary(
        Func<TValue, string> contentSelector,
        int queryChunkSize,
        int maximumCorpusSize,
        RobertaTokenizer tokenizer,
        RobertaModel model,
        ITokenSplitter tokenSplitter,
        string device = "cpu",
        Func<Tensor, Tensor, Tensor> scoreFunction = null)
    {
        this.QueryChunkSize = queryChunkSize;
        this.CorpusChunkSize = queryChunkSize;
        this.MaximumCorpusSize = maximumCorpusSize;

        this.EmbeddingModel = model;
        this.Tokenizer = tokenizer;
        this.TokenSplitter = tokenSplitter;
        this.contentSelector = contentSelector;
        this.scoreFunction = scoreFunction ?? cos_sim;
        this.corpusEmbeddings = torch.zeros(new long[] { this.MaximumCorpusSize, this.EmbeddingSize }, ScalarType.Float32, torch.device(device));
        this.ClearLocked();
    }

    /// <summary>
    /// Gets the size of each query chunk.
    /// </summary>
    public int QueryChunkSize { get; init; }

    /// <summary>
    /// Gets the size of each query chunk.
    /// </summary>
    public int MaximumCorpusSize { get; init; }

    /// <summary>
    /// Gets the size of each query chunk.
    /// </summary>
    public int CorpusChunkSize { get; init; }

    /// <summary>
    /// Gets the maximum number of tokens the model can handle as input.
    /// </summary>
    public int ExcerptSize => this.EmbeddingModel.MaxInputTokenLength;

    public int EmbeddingSize => this.EmbeddingModel.Config.hidden_size;

    /// <summary>
    /// Gets a value indicating whether a re-index is required. This occurs when the fill factor of the corpus drops below 75%.
    /// </summary>
    public bool IsReIndexRequired => this.MaximumCorpusIndex > this.CorpusChunkSize && (this.values.Count / (double)this.MaximumCorpusIndex) < 0.75d;

    public RobertaTokenizer Tokenizer { get; private set; }

    public RobertaModel EmbeddingModel { get; private set; }

    public ITokenSplitter TokenSplitter { get; private set; }

    public int Count => this.values.Count;
    
    /// <summary>
    /// Gets the maximum number of items that have ever been added to the corpus since it was last cleared or re-indexed.
    /// </summary>
    public int MaximumCorpusIndex { get; private set; }

    public bool IsReadOnly => false;

    /// <summary>
    /// Adds or replaces a single item by key (thread-safe).
    /// </summary>
    /// <param name="item">The item to be added to the dictionary.</param>
    public void Add(TKey key, TValue value) => this.Add(new KeyValuePair<TKey, TValue>(key, value));

    /// <summary>
    /// Adds or replaces a single item by key (thread-safe).
    /// </summary>
    /// <param name="item">The item to be added to the dictionary.</param>
    public void Add(KeyValuePair<TKey, TValue> item)
    {
        this.readWrite.EnterWriteLock();
        try
        {
            this.RemoveLocked(item.Key);
            this.MaximumCorpusIndex = this.AppendBatchLocked(new[] { item }, this.MaximumCorpusIndex);
        }
        finally
        {
            this.readWrite.ExitWriteLock();
        }
    }

    /// <summary>
    /// Gets a value indicating whether the key is present in this dictionary (thread-safe).
    /// </summary>
    /// <param name="key">The key to check for.</param>
    /// <returns>True if found, otherwise false.</returns>
    public bool ContainsKey(TKey key)
    {
        this.readWrite.EnterReadLock();
        try
        {
            return this.ContainsKeyLocked(key);
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

    /// <summary>
    /// Removes a single item by key (thread-safe).
    /// </summary>
    /// <param name="item">The item to be added to the dictionary.</param>
    public bool Remove(TKey key)
    {
        this.readWrite.EnterUpgradeableReadLock();
        try
        {
            if (!this.ContainsKeyLocked(key))
            {
                return false;
            }

            this.readWrite.EnterWriteLock();
            try
            {
                return this.RemoveLocked(key);
            }
            finally
            {
                this.readWrite.ExitWriteLock();
            }
        }
        finally
        {
            this.readWrite.ExitUpgradeableReadLock();
        }
    }

    public bool TryGetValue(TKey key, out TValue value)
    {
        this.readWrite.EnterReadLock();
        try
        {
            if (this.values.TryGetValue(key, out var xy))
            {
                value = xy.Value;
                return true;
            }
            
            value = default;
            return false;
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

    public TValue this[TKey key]
    {
        get => this.TryGetValue(key, out var value) ? value : default;
        set => this.Add(key, value);
    }

    public ICollection<TKey> Keys => this.values.Keys;

    public ICollection<TValue> Values
    {
        get
        {
            this.readWrite.EnterReadLock();
            var copy = this.values.Values.Select(kv => kv.Value).ToList();
            this.readWrite.ExitReadLock();
            return copy;
        }
    }

    /// <summary>
    /// Gets an enumerator to the dictionary (WARNING, performs a thread-safe copy of entire list).
    /// </summary>
    /// <returns>The enumerator.</returns>
    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        this.readWrite.EnterReadLock();
        var copy = this.values.Select(kv => new KeyValuePair<TKey,TValue>(kv.Key, kv.Value.Value)).ToList();
        this.readWrite.ExitReadLock();
        return copy.GetEnumerator();
    }

    /// <summary>
    /// Gets an enumerator to the dictionary (not thread-safe).
    /// </summary>
    /// <returns>The enumerator.</returns>
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Clears the entire contents of the dictionary (thread-safe).
    /// </summary>
    public void Clear()
    {
        this.readWrite.EnterWriteLock();
        try
        {
            this.ClearLocked();
        }
        finally
        {
            this.readWrite.ExitWriteLock();
        }
    }

    /// <summary>
    /// Determines if the key of the item is contained by this dictionary (Important does not check the value) (thread-safe).
    /// </summary>
    public bool Contains(KeyValuePair<TKey, TValue> item)
    {
        this.readWrite.EnterReadLock();
        try
        {
            return this.values.TryGetValue(item.Key, out var value) && EqualityComparer<TValue>.Default.Equals(item.Value, value.Value);
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

    /// <summary>
    /// Copies the entire contents of the dictionary to part of a target array (thread-safe).
    /// </summary>
    /// <param name="array"></param>
    /// <param name="arrayIndex"></param>
    public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
    {
        this.readWrite.EnterReadLock();
        try
        {
            ((ICollection<KeyValuePair<TKey, TValue>>)this.values).CopyTo(array, arrayIndex);
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

    /// <summary>
    /// Removes the key on the item (thread-safe).
    /// </summary>
    /// <param name="item"></param>
    /// <returns></returns>
    public bool Remove(KeyValuePair<TKey, TValue> item) => this.Remove(item.Key);

    /// <summary>
    /// Adds or replaces a single item by key (thread-safe).
    /// </summary>
    /// <param name="item">The item to be added to the dictionary.</param>
    public void Add((TKey Key, TValue Value) item) => this.Add(new KeyValuePair<TKey, TValue>(item.Key, item.Value));

    /// <summary>
    /// Loads the tokenizer and the model for creating embeddings.
    /// </summary>
    /// <returns>Task for completion.</returns>
    public static async Task<SemanticDictionary<TKey, TValue>> CreateAsync(
        Func<TValue, string> contentSelector,
        int queryChunkSize = 1000,
        int maximumCorpusSize = 500000,
        string embeddingModelName = "all-distilroberta-v1",
        string device = "cpu")
    {
        var tokenizer = await RobertaTokenizer.from_pretrained(embeddingModelName, device);
        var embeddingModel = await RobertaModel.from_pretrained(embeddingModelName, device);
        var tokenSplitter = new SentenceSplitter(embeddingModel.MaxInputTokenLength);
        return new SemanticDictionary<TKey, TValue>(contentSelector, queryChunkSize, maximumCorpusSize, tokenizer, embeddingModel, tokenSplitter, device);
    }

    /// <summary>
    /// Adds or replaces a set of items by key.
    /// </summary>
    /// <param name="items">The items to be indexed semantically.</param>
    public void AddAll(IEnumerable<KeyValuePair<TKey, TValue>> items)
    {
        foreach (var page in items.Paginate(this.QueryChunkSize))
        {
            this.readWrite.EnterWriteLock();
            try
            {
                foreach (var item in items)
                {
                    this.RemoveLocked(item.Key);
                }

                this.MaximumCorpusIndex = this.AppendBatchLocked(page, this.MaximumCorpusIndex);
            }
            finally
            {
                this.readWrite.ExitWriteLock();
            }
        }
    }

    /// <summary>
    /// Appends a batch of items at once, assuming thread lock for Write access has already been taken.
    /// </summary>
    /// <param name="items">A batch of items.</param>
    /// <param name="startOffset">The starting offset to add or replace the corpus embeddings from</param>
    /// <returns>The new starting offset in the corpus.</returns>
    private int AppendBatchLocked(IReadOnlyList<KeyValuePair<TKey, TValue>> items, int startOffset)
    {
        int start = startOffset;
        int end = start;

        // IMPROVE: AC: We could probably embed more sentences at once (across source documents) which would be better for GPU based indexing.
        for (int i = 0; i < items.Count; i++)
        {
            var token_chunks = this.TokenSplitter.Split(this.Tokenizer.Encode(this.contentSelector(items[i].Value))).ToList();
            if (token_chunks.Count > 0)
            {
                var encoded_input = RobertaTokenizer.ToTensor(token_chunks);
                var embeddings = this.EmbeddingModel.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);
                end = start + token_chunks.Count;
                this.corpusEmbeddings[start..end].copy_(embeddings);
                this.corpusKeyRanges.Add((start, end - 1, items[i].Key, token_chunks));
                this.values[items[i].Key] = (start..end, items[i].Value);
                start = end;
            }
        }

        return end;
    }

    public async Task AddAllAsync(IAsyncEnumerable<KeyValuePair<TKey, TValue>> sentences, CancellationToken cancellation)
    {
        await foreach (var page in sentences.Paginate(this.QueryChunkSize, cancellation))
        {
            try
            {
                this.readWrite.EnterWriteLock();
                foreach (var (k, v) in page)
                {
                    this.RemoveLocked(k);
                }

                this.MaximumCorpusIndex = this.AppendBatchLocked(page, this.MaximumCorpusIndex);
            }
            finally
            {
                this.readWrite.ExitWriteLock();
            }
        }
    }

    private bool RemoveLocked(TKey key)
    {
        var (range, value) = this.values.GetValueOrDefault(key);
        if (this.values.Remove(key))
        {
            var (offset, length) = range.GetOffsetAndLength(this.MaximumCorpusSize);
            if (length > 0)
            {
                // AC: A zero embedding is effectively removed as a torch.mm will return 0 similarity with all vectors.
                this.corpusEmbeddings[offset..(offset + length), ..].zero_();
                var index = this.IndexOfCorpusKeyRange(offset);
                if (index != -1)
                {
                    this.corpusKeyRanges.RemoveAt(index);
                }
            }

            return true;
        }

        return false;
    }

    /// <summary>
    /// Gets the top N search results for each of the provided sentences.
    /// </summary>
    /// <param name="sentences"></param>
    /// <param name="topK"></param>
    /// <returns></returns>
    public IReadOnlyList<IReadOnlyList<SearchResult<TKey, TValue>>> GetBatchTop(IReadOnlyList<string> sentences, int topK = 10)
    {
        var encoded_input = this.Tokenizer.Tokenize(sentences);
        var embeddings = this.EmbeddingModel.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);
        this.readWrite.EnterReadLock();
        try
        {
            return this.GetBatchTopLocked(embeddings, topK).ToList();
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

    /// <summary>
    /// Gets the top N most similar results in batches.
    /// This method assumes the thread lock is already taken for read access.
    /// </summary>
    /// <param name="queryEmbeddings">All query embeddings in the shape of (Batch, Embedding).</param>
    /// <param name="topK">The top results per batch to return (default is 10).</param>
    /// <returns>A list of batches of search results.</returns>
    private IEnumerable<IReadOnlyList<SearchResult<TKey, TValue>>> GetBatchTopLocked(Tensor queryEmbeddings, int topK = 10)
    {
        if (queryEmbeddings.dim() == 1)
        {
            queryEmbeddings = queryEmbeddings.unsqueeze(0);
        }

        if (corpusEmbeddings.device.type != queryEmbeddings.device.type)
        {
            queryEmbeddings = queryEmbeddings.to(corpusEmbeddings.device);
        }

        // Iterate through the pages of query sentences
        for (int queryStartIdx = 0; queryStartIdx < queryEmbeddings.shape[0]; queryStartIdx += this.QueryChunkSize)
        {
            var results = new SortedList<float, SearchResult<TKey, TValue>>(topK, ScoreComparer);

            // Iterate through the corpus in chunks.
            for (int corpusStartIdx = 0; corpusStartIdx < this.MaximumCorpusIndex; corpusStartIdx += this.CorpusChunkSize)
            {
                var corpusLastIdx = Math.Min(this.MaximumCorpusIndex, corpusStartIdx + this.CorpusChunkSize);
                var cosScores = scoreFunction(queryEmbeddings[queryStartIdx..(queryStartIdx + this.QueryChunkSize)], corpusEmbeddings[corpusStartIdx..corpusLastIdx]);
                var (cosScoresTopKValues, cosScoresTopKIdx) = torch.topk(cosScores, Math.Min(topK, (int)cosScores.shape[1]), dim: 1, largest: true, sorted: false);

                for (int r = 0; r < cosScores.shape[0]; r++)
                {
                    for (int i = 0; i < cosScoresTopKIdx[r].shape[0]; i++)
                    {
                        int corpusId = corpusStartIdx + (int)cosScoresTopKIdx[r, i].ToInt32();
                        int queryId = queryStartIdx + r;

                        var score = cosScoresTopKValues[r, i].ToSingle();
                        if (score > 0)
                        {
                            results.Add(score, TranslateScore(corpusId, score));
                            if (results.Count > topK)
                            {
                                results.RemoveAt(topK);
                            }
                        }
                    }
                }
            }

            yield return (IReadOnlyList<SearchResult<TKey, TValue>>)results.Values.ToList();
        }
    }

    private static Tensor cos_sim(Tensor a, Tensor b)
    {
        if (a.shape.Length == 1)
        {
            a = a.unsqueeze(0);
        }

        if (b.shape.Length == 1)
        {
            b = b.unsqueeze(0);
        }

        // AC: Sentence_embeddings already normalizes the vectors.
        ////var a_norm = a.normalize(p: 2, dim: 1);
        ////var b_norm = b.normalize(p: 2, dim: 1);
        return torch.mm(a, b.transpose(0, 1));
    }

    private IEnumerable<SearchResult<TKey, TValue>> TranslateScores(IEnumerable<(int Index, float Score)> scores) => scores.Select(k => TranslateScore(k.Index, k.Score));

    private SearchResult<TKey, TValue> TranslateScore(int index, float score)
    {
        var i = this.IndexOfCorpusKeyRange(index);
        var (start, end, key, excerpts) = this.corpusKeyRanges[i];
        return new SearchResult<TKey, TValue>(key, this.values.GetValueOrDefault(key).Value, this.Tokenizer.Decode(excerpts[index - start]), score);
    }

    private void ClearLocked()
    {
        if (this.MaximumCorpusIndex == 0)
        {
            return;
        }

        this.corpusEmbeddings.zero_();
        this.MaximumCorpusIndex = 0;
        this.corpusKeyRanges.Clear();
        this.values.Clear();
    }

    private bool ContainsKeyLocked(TKey key) => this.values.ContainsKey(key);

    /// <summary>
    /// Determines the start of the range of values for a given offset, or -1 if no range exists.
    /// </summary>
    /// <param name="corpusIndex"></param>
    /// <returns></returns>
    private int IndexOfCorpusKeyRange(int corpusIndex)
    {
        var x = this.corpusKeyRanges.BinarySearch((corpusIndex, corpusIndex, default, null), CorpusRangeComparer);

        // AC: If not found, use twos-complement to find closest match.
        return x < -1 ? (~x) - 1 : x;
    }
    
    private class SearchResultScoreComparer : IComparer<SearchResult<TKey, TValue>>, IComparer<float>
    {
        public int Compare(float x, float y) => y.CompareTo(x);
        public int Compare(SearchResult<TKey, TValue>? x, SearchResult<TKey, TValue>? y) => y.Score.CompareTo(x.Score);
    }
    
    private class CorpusKeyRangeComparer : IComparer<(int StartIndex, int EndIndex, TKey Key, List<IReadOnlyList<int>> Excerpts)>
    {
        public int Compare((int StartIndex, int EndIndex, TKey Key, List<IReadOnlyList<int>> Excerpts) x, (int StartIndex, int EndIndex, TKey Key, List<IReadOnlyList<int>> Excerpts) y)
        {
            if (x.StartIndex <= y.EndIndex && x.EndIndex >= y.StartIndex)
            {
                return 0;
            }
            
            return x.StartIndex.CompareTo(y.StartIndex);
        }
    }
}