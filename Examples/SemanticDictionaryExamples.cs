using System.Diagnostics;

namespace PerceptivePyro.Examples;

internal class SemanticDictionaryExamples
{
    /// <summary>
    /// Indexes a bunch of sentences into a dictionary and then looks up from a similar sentence.
    /// </summary>
    /// <returns></returns>
    public static async Task Semantic_Dictionary_TopK()
    {
        var dict = await SemanticDictionary<string, string>.CreateAsync(k => k);

        dict.AddAll(new KeyValuePair<string, string>[]
        {
            new("1", "The morning train leaves at 12:00pm"),
            new("2", "Frank is working from home today"),
            new("3", "Traditionally the builder events were handled via HandleAsync, however a newer static builder type was more recently added that supports static Handle() methods which are much more efficient, so they are now the preferred method of writing a builder. The HandleAsync methods are there for backwards compatibility (and of course if you needed to do something asynchronous in the builder).")
        });

        var s = Stopwatch.StartNew();
        var questions = new[]
        {
            "What time does the morning train leave?",
            "Is someone able to give me a brief explanation of why some builders use Handle() and some use  HandleAsync()  please?"
        };
        var answers = dict.GetBatchTop(questions);
        s.ElapsedMilliseconds.Dump();
        questions.Zip(answers).ToList().Dump();

        // Remove from a vector database!
        dict.Remove("3");
        var answers2 = dict.GetBatchTop(questions);
        answers2.Dump();
    }

    public static async Task Semantic_Dictionary_Sentence_Splitting()
    {
        var dict = await SemanticDictionary<string, string>.CreateAsync(k => k);
        dict.Add(
            "Document A",
            """"
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nibh ex, convallis ac elit a, varius viverra lorem. Mauris nisl sem, maximus et tortor at, bibendum mollis orci. Nullam imperdiet leo a augue viverra interdum. Nullam vel ligula maximus mi blandit bibendum. Vivamus dictum congue nisl ut varius. Cras in sem lacus. Ut sed gravida turpis.

            Nulla a mi ullamcorper, semper dui at, pharetra nisl. Suspendisse fringilla blandit dolor, id scelerisque nibh ultricies eget. In id vulputate leo, ut posuere eros. Donec interdum lobortis ex ut tincidunt. Nunc non nibh at lorem aliquet scelerisque et eget odio. Nunc id elit a urna aliquet viverra id vel quam. Proin eget justo ultrices leo tristique fermentum sed eu eros. Proin posuere eros nec nunc ornare tincidunt. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Donec consequat, nisl eget feugiat accumsan, nulla est lacinia dolor, et aliquet lacus sapien et odio. Phasellus tristique bibendum semper. In augue metus, euismod ac rhoncus vel, eleifend in sapien. Phasellus dictum consequat auctor.

            Fusce eu sodales tellus. Nulla facilisi. Integer mattis tellus vel sapien rhoncus, eget consectetur turpis euismod. Mauris a ex eget quam iaculis posuere. Vestibulum sem eros, mollis vitae lacinia at, commodo at lectus. Donec ac dolor consequat, viverra sapien in, ultricies tellus. Sed purus elit, porttitor sit amet suscipit vitae, sagittis in ante. Duis nibh tortor, sagittis non luctus non, blandit et nisl.

            Aliquam erat volutpat. Aliquam posuere rutrum erat, nec dictum massa. Quisque dolor nunc, fermentum vitae dictum in, aliquam mollis turpis. Etiam sed magna porta nibh dapibus vestibulum. Sed posuere finibus mi, sit amet pretium velit tempor a. Donec tempus, nibh eget consequat tristique, diam nibh accumsan ligula, at placerat enim dolor eu tellus. Sed id metus vitae quam sagittis volutpat interdum nec ante. Donec eu rutrum metus. Maecenas feugiat ultrices pretium. Nulla suscipit metus metus, ut gravida ligula consequat eget. In hac habitasse platea dictumst.

            Aliquam pretium tempus ultricies. Pellentesque vel dignissim lacus, et vehicula leo. Donec vitae dictum nunc, vel facilisis erat. Ut at lectus quis nunc bibendum iaculis. Vestibulum tincidunt suscipit vehicula. Suspendisse commodo neque sit amet commodo lacinia. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Curabitur in nibh at justo fringilla volutpat. Pellentesque ornare purus nibh, vitae pulvinar diam porta eget. Quisque non eleifend dui. Etiam at tempor nibh. Curabitur placerat metus eu dui elementum accumsan. In gravida dui quis ante ullamcorper fermentum. Donec eu metus vitae diam condimentum imperdiet vel ac est.

            Donec malesuada id urna at laoreet. Nam mi tortor, ornare nec velit in, vestibulum fringilla nulla. Vestibulum aliquam mauris fringilla orci volutpat dictum nec quis metus. Praesent molestie maximus dui id volutpat. Aliquam scelerisque lorem nunc, eget interdum lorem interdum a. Aliquam feugiat urna nec facilisis luctus. Nullam elementum neque et egestas molestie. Praesent id magna non ex facilisis dapibus. Duis eget justo nunc. Nunc vehicula nisi nec ante consequat, ut scelerisque nisi congue. Maecenas felis massa, laoreet in faucibus in, ornare sit amet ligula. Sed feugiat sapien eget accumsan hendrerit. Maecenas dapibus id enim at dignissim. Vivamus dictum et tellus vel viverra. Cras finibus suscipit cursus.

            Cras ullamcorper rutrum dui, eget iaculis lacus ornare sit amet. Aenean cursus suscipit nulla in lobortis. Ut ut volutpat metus. Vivamus at suscipit ligula, vitae placerat lectus. Praesent non ligula eget elit egestas tempus eu id orci. Aliquam libero purus, mollis sed tristique et, consequat non mi. Cras nec ex at est euismod aliquam.

            Nulla vitae tincidunt arcu, quis efficitur tellus. Nulla vehicula dui sit amet nisi malesuada, et dapibus libero blandit. Morbi tincidunt semper ex sit amet laoreet. Nunc eget lorem vitae justo mattis condimentum ut in ex. Morbi in lorem in ex vestibulum tempor. Ut ac cursus purus. Praesent nec varius libero. Pellentesque magna arcu, bibendum at eleifend non, bibendum sit amet leo. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Suspendisse ac ornare sem. Duis diam tortor, molestie nec ullamcorper sed, laoreet vel mauris.
            """");
        
        var answers = dict.GetBatchTop(new[] { "Nulla a mi ullamcorper" }, 2);
        answers.Dump();
    }
}