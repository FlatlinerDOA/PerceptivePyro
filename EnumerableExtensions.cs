using System.Runtime.CompilerServices;

namespace PerceptivePyro;

public static class EnumerableExtensions
{
    public static IEnumerable<IReadOnlyList<T>> Paginate<T>(this IEnumerable<T> source, int pageSize)
    {
        if (pageSize <= 0)
        {
            throw new ArgumentException("pageSize must be greater than 0");
        }

        List<T> page = new List<T>(pageSize);
        foreach (T item in source)
        {
            page.Add(item);

            if (page.Count == pageSize)
            {
                yield return page;
                page = new List<T>(pageSize);
            }
        }

        if (page.Count > 0)
        {
            yield return page;
        }
    }
    
    public static async IAsyncEnumerable<IReadOnlyList<T>> Paginate<T>(this IAsyncEnumerable<T> source, int pageSize, [EnumeratorCancellation] CancellationToken cancellation)
    {
        if (pageSize <= 0)
        {
            throw new ArgumentException("pageSize must be greater than 0");
        }

        List<T> page = new List<T>(pageSize);
        await foreach (T item in source)
        {
            page.Add(item);

            if (page.Count == pageSize)
            {
                yield return page;
                page = new List<T>(pageSize);
            }
        }

        if (page.Count > 0)
        {
            yield return page;
        }
    }
}