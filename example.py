from keyrelbert import KeyRelBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
      """
keyphrase_extractor = KeyRelBERT()
keyphrase_ngram_range=(1, 2)
doc_embeddings, word_embeddings = keyphrase_extractor.extract_embeddings(doc, keyphrase_ngram_range=keyphrase_ngram_range)
kw_score_list = keyphrase_extractor.extract_keywords(doc, 
    keyphrase_ngram_range=keyphrase_ngram_range,
    doc_embeddings=doc_embeddings, 
    word_embeddings=word_embeddings, 
    use_mmr=True,
    top_n=10,
    diversity=0.6,
)
relations = keyphrase_extractor.extract_relations(doc, 
    keywords=kw_score_list, 
    doc_embeddings=doc_embeddings, 
    word_embeddings=word_embeddings,
    keyphrase_ngram_range=keyphrase_ngram_range,
    )