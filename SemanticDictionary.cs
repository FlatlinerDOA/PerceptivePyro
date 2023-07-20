namespace PerceptivePyro;

public sealed class SemanticDictionary<TKey, TValue> : IDictionary<TKey, TValue>
{
    private readonly ReaderWriterLockSlim readWrite = new ReaderWriterLockSlim(LockRecursionPolicy.NoRecursion);
    private readonly Func<TValue, string> contentSelector;
    private readonly Func<Tensor,Tensor,Tensor> scoreFunction;

    private Dictionary<int, TKey> offsetKeys;
    private Dictionary<TKey, int> keyOffsets;
    private Dictionary<TKey, TValue> values;
    private Tensor corpusEmbeddings;
    private RobertaTokenizer tokenizer;
    private RobertaModel model;

    public SemanticDictionary(Func<TValue, string> contentSelector, int queryChunkSize = 1000, int corpusChunkSize = 500000, string embeddingModel = "all-distilroberta-v1", Func<Tensor, Tensor, Tensor> scoreFunction = null)
    {
        this.QueryChunkSize = queryChunkSize;
        this.CorpusChunkSize = corpusChunkSize;
        this.EmbeddingModel = embeddingModel;
        
        this.contentSelector = contentSelector;
        this.scoreFunction = scoreFunction;
        this.scoreFunction ??= cos_sim;
        this.corpusEmbeddings = torch.zeros(new long[] { this.CorpusChunkSize, this.EmbeddingSize }, ScalarType.Float32);
        this.ClearLocked();
    }

    public int QueryChunkSize { get; init; }

    public string EmbeddingModel { get; init; }

    public int CorpusChunkSize { get; init; }
    
    public int ExcerptSize => 512;
    
    public int EmbeddingSize => 768;

    public int Count => this.values.Count;
    
    public bool IsReadOnly => false;

    public async Task InitializeAsync()
    {
        this.readWrite.EnterWriteLock();
        try
        {
            this.ClearLocked();
            this.tokenizer = await RobertaTokenizer.from_pretrained(this.EmbeddingModel);
            this.model = await RobertaModel.from_pretrained(this.EmbeddingModel);
        }
        finally
        {
            this.readWrite.ExitWriteLock();            
        }
    }
    
    public void AddAll(IEnumerable<KeyValuePair<TKey, TValue>> sentences)
    {
        foreach (var page in sentences.Paginate(this.QueryChunkSize))
        {
            this.readWrite.EnterWriteLock();
            try
            {
                var startOffset = this.values.Count;
                var end = AddPageLocked(page, startOffset);
                startOffset = this.values.Count;
            }
            finally
            {
                this.readWrite.ExitWriteLock();
            }
        }
    }

    private int AddPageLocked(IReadOnlyList<KeyValuePair<TKey, TValue>> page, int startOffset)
    {
        var encoded_input = tokenizer.Tokenize(page.Select(s => this.contentSelector(s.Value)).ToList());
        var embeddings = model.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);
        int start = startOffset * this.QueryChunkSize;
        int end = start + page.Count;
        this.corpusEmbeddings[start..end].copy_(embeddings);
        for (int i = 0; i < page.Count; i++)
        {
            this.offsetKeys[start + i] = page[i].Key;
            this.keyOffsets[page[i].Key] = start + i;
            this.values[page[i].Key] = page[i].Value;
        }

        return end;
    }

    public async Task AddAllAsync(IAsyncEnumerable<KeyValuePair<TKey, TValue>> sentences, CancellationToken cancellation)
    {
        this.readWrite.EnterWriteLock();
        try
        {
            var startOffset = this.values.Count;
            await foreach (var page in sentences.Paginate(this.QueryChunkSize, cancellation))
            {
                var end = this.AddPageLocked(page, startOffset);
                startOffset = end;
            }
        }
        finally
        {
            this.readWrite.ExitWriteLock();
        }
    }

    public void Add(TKey key, TValue value)
    {
        this.readWrite.EnterWriteLock();
        try
        {
            this.RemoveLocked(key);
            var startOffset = this.values.Count;
            var encoded_input = tokenizer.Tokenize(new[] { this.contentSelector(value) });
            var embeddings = model.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);
            int start = startOffset;
            int end = start + 1;
            this.corpusEmbeddings[start..end].copy_(embeddings);
            this.offsetKeys[start] = key;
            this.values[key] = value;
            startOffset = end;
        }
        finally
        {
            this.readWrite.ExitWriteLock();
        }
    }
    
    public void Add(KeyValuePair<TKey, TValue> item) => this.Add(item.Key, item.Value);

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
    
    private bool RemoveLocked(TKey key)
    {
        if (this.values.Remove(key))
        {
            var offset = this.keyOffsets.GetValueOrDefault(key, -1);
            if (offset != -1)
            {
                // AC: A zero embedding is effectively removed as a torch.mm will return 0 similarity with all vectors.
                this.corpusEmbeddings[offset..(offset + 1), ..].zero_();
                this.keyOffsets.Remove(key);
                this.offsetKeys.Remove(offset);
            }
            
            return true;
        }
        
        return false;
    }

    public bool TryGetValue(TKey key, out TValue value)
    {
        this.readWrite.EnterReadLock();
        try
        {
            return this.values.TryGetValue(key, out value);
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

    public ICollection<TKey> Keys { get; set; }
    public ICollection<TValue> Values { get; set; }

    public List<List<SearchResult<TKey, TValue>>> GetBatchTop(string[] sentences, int topK = 10)
    {
        var encoded_input = tokenizer.Tokenize(sentences);
        var embeddings = model.sentence_embeddings(encoded_input.input_ids, encoded_input.attention_mask);
        this.readWrite.EnterReadLock();
        try
        {
            return this.GetBatchTopLocked(embeddings, topK);
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }
    
    private List<List<SearchResult<TKey, TValue>>> GetBatchTopLocked(Tensor queryEmbeddings, int topK = 10)
    {
        if (queryEmbeddings.dim() == 1)
        {
            queryEmbeddings = queryEmbeddings.unsqueeze(0);
        }

        if (corpusEmbeddings.device.type != queryEmbeddings.device.type)
        {
            queryEmbeddings = queryEmbeddings.to(corpusEmbeddings.device);
        }

        List<List<SearchResult<TKey, TValue>>> queriesResultList = Enumerable.Repeat(0, (int)queryEmbeddings.shape[0]).Select(x => new List<SearchResult<TKey, TValue>>(topK)).ToList();

        for (int queryStartIdx = 0; queryStartIdx < queryEmbeddings.shape[0]; queryStartIdx += this.QueryChunkSize)
        {
            for (int corpusStartIdx = 0; corpusStartIdx < corpusEmbeddings.shape[0]; corpusStartIdx += this.CorpusChunkSize)
            {
                Tensor cosScores = scoreFunction(queryEmbeddings[queryStartIdx..(queryStartIdx+this.QueryChunkSize)], corpusEmbeddings[corpusStartIdx..(corpusStartIdx+this.CorpusChunkSize)]);
                (Tensor cosScoresTopKValues, Tensor cosScoresTopKIdx) = torch.topk(cosScores, Math.Min(topK, (int)cosScores.shape[1]), dim: 1, largest: true, sorted: false);

                for (int queryItr = 0; queryItr < cosScores.shape[0]; queryItr++)
                {
                    for (int i = 0; i < cosScoresTopKIdx[queryItr].shape[0]; i++)
                    {
                        int corpusId = corpusStartIdx + (int)cosScoresTopKIdx[queryItr, i].ToInt32();
                        int queryId = queryStartIdx + queryItr;
                        
                        var score = cosScoresTopKValues[queryItr, i].ToSingle();
                        if (score != 0)
                        {
                            if (queriesResultList[queryId].Count < topK)
                            {
                                queriesResultList[queryId].Add(TranslateScore(corpusId, score));
                            }
                            else
                            {
                                queriesResultList[queryId].Sort((x, y) => y.Score.CompareTo(x.Score));
                                if (queriesResultList[queryId][topK - 1].Score < score)
                                {
                                    queriesResultList[queryId][topK - 1] = TranslateScore(corpusId, score);
                                }
                            }
                        }
                    }
                }
            }
        }

        foreach (List<SearchResult<TKey, TValue>> resultList in queriesResultList)
        {
            resultList.Sort((x, y) => y.Score.CompareTo(x.Score));
        }

        return queriesResultList;
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
        var key = this.offsetKeys.GetValueOrDefault(index);
        return new SearchResult<TKey, TValue>(key, this.values.GetValueOrDefault(key), score);
    }
    
    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => this.values.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

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

    private void ClearLocked()
    {
        if (this.values != null && this.values.Count == 0)
        {
            return;
        }

        this.corpusEmbeddings.zero_();
        this.offsetKeys = new Dictionary<int, TKey>();
        this.keyOffsets = new Dictionary<TKey, int>();
        this.values = new Dictionary<TKey, TValue>();
    }

    public bool Contains(KeyValuePair<TKey, TValue> item)
    {
        this.readWrite.EnterReadLock();
        try
        {
            return this.values.TryGetValue(item.Key, out var value) && EqualityComparer<TValue>.Default.Equals(item.Value, value);
        }
        finally
        {
            this.readWrite.ExitReadLock();
        }
    }

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

    private bool ContainsKeyLocked(TKey key) => this.values.ContainsKey(key);

    public bool Remove(KeyValuePair<TKey, TValue> item) => this.Remove(item.Key);
}

public sealed record SearchResult<TKey, TValue>(TKey Key, TValue Value, float Score);