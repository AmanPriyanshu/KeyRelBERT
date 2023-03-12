from keyrelbert import KeyRelBERT

doc = """
         Billy Mays, the bearded, boisterious pitchman who, as the undisputed king of TV yell and sell, became an inlikely pop culture icon, died at his home in Tampa, Fla, on Sunday.
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
print(relations)