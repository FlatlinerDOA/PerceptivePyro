namespace NanoGPTSharp.VectorSearch;

public sealed record IndexedVector(int Id, float[] Values);

public sealed class BTreeNode : IEquatable<BTreeNode>
{
    private float? norm;

    public BTreeNode(IndexedVector point)
    {
        this.Point = point;
    }

    public IndexedVector Point { get; init; }

    public BTreeNode? Left { get; set; }

    public BTreeNode? Right { get; set; }

    public bool IsLeaf => this.Left == null && this.Right == null;

    public float Norm => this.norm ??= this.Point.Values.Norm();

    public bool Equals(BTreeNode? other) => this.Point.Id == (other?.Point.Id ?? -1);
}


public sealed class EmbeddingIndex
{
    // bigger to top
    private static readonly Comparer<float> HighestToLowestComparer = Comparer<float>.Create((x, y) => y.CompareTo(x)); 

    // Editable sequence of root nodes to start searching from.
    ////private Rope<BTreeNode> roots = new Rope<BTreeNode>();

    private int maxDepth;

    private readonly DistanceMetric distanceMetric;

    public EmbeddingIndex(int maxDepth, DistanceMetric distanceMetric)
    {
        this.maxDepth = maxDepth;
        this.distanceMetric = distanceMetric;

        // Initialize an empty
        ////this.roots = new Rope<BTreeNode>(); 

        // Proposed algorithm:
        // 1. Iteratively add to the last cluster, The cluster is kept sorted by distance from the origin.
        // 2. Kd tree is built for 
        // 2. Once a rope gets long enough (as configured by maxDepth), add another rope.
        
        // Things to note:
        // Ropes maintain their balance automatically.
        // Ropes are immutable and so we naturally have a thread-safe index.
    }

    public void Insert(IndexedVector point)
    {
        var nodeToAddTo = this.roots.Length > 0 ? this.roots[this.roots.Length - 1] : null;
        this.roots = Insert(nodeToAddTo, new BTreeNode(point), 0);
    }

    private BTreeNode Insert(BTreeNode? node, BTreeNode point, int depth)
    {
        if (node == null)
        {
            node = point;
            return node;
        }

        var isLeft = Distance(node.Point, point.Point) >= 0;
        if (isLeft)
        {
            node.Left = Insert(node.Left, point, depth + 1);
        }
        else
        {
            node.Right = Insert(node.Right, point, depth + 1);
        }

        return node;
    }

    public IEnumerable<IndexedVector> SearchKNN(float[] query, int top = 10)
    {

    }

    private SearchResult SearchKNN1(BTreeNode root, BTreeNode? current, float[] query, SearchResult best)
    {
        if (current == null)
        {
            return best;
        }

        var current_distance = this.distanceMetric switch
        {
            DistanceMetric.Euclidean => current.Point.Values.Zip(query).EuclideanMargin(root.Norm),
            DistanceMetric.Cosine => current.Point.Values.Zip(query).CosineMargin()
        }; current.Point.Values.Zip(query).EuclideanMargin(current.Point.Values[0]);
        if (current.IsLeaf)
        {
            return best == null || current_distance < best.distance ? new SearchResult(current, current_distance) : best;
        }
        
        var isLeft = current_distance >= 0;
        var result = isLeft ? this.SearchKNN1(root, current.Left, query, best) : this.SearchKNN1(root, current.Right, query, best);
        return result.distance < best.distance ? result : best;
    }

    private SearchResult GetNearest(float[] queryVector, int nResults, int limitTrees = -1, int search_k = -1)
    {
        var pq = new PriorityQueue<BTreeNode, float>(HighestToLowestComparer);
        // PriorityQueue size: roots.Count() * FLOAT_SIZE);
        const float kMaxPriority = 1e30f;
        var useRoots = this.roots;
        if (limitTrees > 0 && limitTrees < this.roots.Length)
        {
            useRoots = this.roots.Slice(0, limitTrees);
        }

        if (search_k <= 0)
        {
            search_k = useRoots.Length * nResults;
        }

        foreach (var r in useRoots)
        {
            pq.Enqueue(r, kMaxPriority); // add(new PQEntry(kMaxPriority, r));
        }

        var nearestNeighbors = new HashSet<IndexedVector>();
        while (nearestNeighbors.Count() < search_k && pq.Count != 0)
        {
            var topNodeOffset = pq.Dequeue(); //  top; //.nodeOffset;
            int nDescendants = topNodeOffset.IsLeaf;
            float[] v = getNodeVector(topNodeOffset);
            if (nDescendants == 1)
            {  // n_descendants
               // (from Java) FIXME: does this ever happen?
                if (v.IsZeroVec())
                {
                    continue;
                }

                nearestNeighbors.Add(topNodeOffset / NODE_SIZE);

            }
            else if (nDescendants <= MIN_LEAF_SIZE)
            {

                for (int i = 0; i < nDescendants; i++)
                {
                    int j = GetInt(topNodeOffset + INDEX_TYPE_OFFSET + i * INT_SIZE);
                    if (isZeroVec(getNodeVector(j * NODE_SIZE)))
                        continue;
                    nearestNeighbors.Add(j);
                }

            }
            else
            {

                float margin = (INDEX_TYPE == IndexType.ANGULAR) ?
                        cosineMargin(v, queryVector) :
                        euclideanMargin(v, queryVector, getNodeBias(topNodeOffset));
                long childrenMemOffset = topNodeOffset + INDEX_TYPE_OFFSET;
                long lChild = NODE_SIZE * (long)GetUInt(childrenMemOffset);
                long rChild = NODE_SIZE * (long)GetUInt(childrenMemOffset + 4);
                pq.Enqueue(lChild, -margin);
                pq.Enqueue(rChild, margin);
            }
        }

        //SimplePriorityQueue<int> sortedNNs = new SimplePriorityQueue<int>(); // reverseComparer
        var sortedNNs = new PriorityQueue<int, float>(reverseComparer); // reverseComparer
                                                                        //List<PQEntry> sortedNNs = new List<PQEntry>();

        foreach (int nn in nearestNeighbors)
        {
            float[] v = GetItemVector(nn);
            if (v.IsZeroVec()) {
                continue;
            }
            float priority = this.distanceMetric == DistanceMetric.Cosine ? 
                    v.Zip(queryVector).CosineMargin() :
                    -v.Zip(queryVector).EuclideanDistance();
            sortedNNs.Enqueue(nn, priority);
        }

        return sortedNNs.Take(nResults).ToArray();
    }

    /// <summary>
    /// Returns a non-negative number identifying the distance between two points in N-dimensional vector space.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    private float Distance(IndexedVector a, IndexedVector b) => this.distanceMetric switch
    {
        DistanceMetric.Euclidean => a.Values.Zip(b.Values).EuclideanMargin(a.Norm),
        DistanceMetric.Cosine => a.Values.Zip(b.Values).CosineMargin()
    };

    private record SearchResult(BTreeNode? node = null, float distance = float.MaxValue);
}
