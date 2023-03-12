# KeyRelBERT
KeyRelBERT is a streamlined pipeline for keyword extraction and relation extraction that leverages SentenceTransformers embeddings to expand keyword extraction & employs unsupervised similarity computation for relation extraction.

## Usage

The most minimal example can be seen below for the extraction of keywords:
```python
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
```


## Extracted Relations:

```
{'billy mays:|:pitchman undisputed': 'cast member', 'pitchman undisputed:|:billy mays': 'original network', 'billy mays:|:icon died': 'date of death', 'icon died:|:billy mays': 'date of death', 'billy mays:|:bearded boisterious': 'mouth of the watercourse', 'bearded boisterious:|:billy mays': 'mouth of the watercourse', 'billy mays:|:king tv': 'cast member', 'king tv:|:billy mays': 'original network', 'billy mays:|:pop culture': 'influenced by', 'pop culture:|:billy mays': 'genre', 'billy mays:|:yell sell': 'cast member', 'yell sell:|:billy mays': 'capital', 'billy mays:|:inlikely': 'residence', 'inlikely:|:billy mays': 'residence', 'billy mays:|:tampa fla': 'headquarters location', 'tampa fla:|:billy mays': 'located in or next to body of water', 'billy mays:|:sunday': 'cast member', 'sunday:|:billy mays': 'city of residence', 'pitchman undisputed:|:icon died': 'followed by', 'icon died:|:pitchman undisputed': 'cause of death', 'pitchman undisputed:|:bearded boisterious': 'performer', 'bearded boisterious:|:pitchman undisputed': 'subsidiary', 'pitchman undisputed:|:king tv': 'original language of work', 'king tv:|:pitchman undisputed': 'original language of work', 'pitchman undisputed:|:pop culture': 'top members', 'pop culture:|:pitchman undisputed': 'genre', 'pitchman undisputed:|:yell sell': 'major shareholder', 'yell sell:|:pitchman undisputed': 'followed by', 'pitchman undisputed:|:inlikely': 'headquarters location', 'inlikely:|:pitchman undisputed': 'followed by', 'pitchman undisputed:|:tampa fla': 'major shareholder', 'tampa fla:|:pitchman undisputed': 'production company', 'pitchman undisputed:|:sunday': 'performer', 'sunday:|:pitchman undisputed': 'publication date', 'icon died:|:bearded boisterious': 'cause of death', 'bearded boisterious:|:icon died': 'mouth of the watercourse', 'icon died:|:king tv': 'cause of death', 'king tv:|:icon died': 'series', 'icon died:|:pop culture': 'date of death', 'pop culture:|:icon died': 'genre', 'icon died:|:yell sell': 'replaced by', 'yell sell:|:icon died': 'followed by', 'icon died:|:inlikely': 'cause of death', 'inlikely:|:icon died': 'instance of', 'icon died:|:tampa fla': 'date of death', 'tampa fla:|:icon died': 'location', 'icon died:|:sunday': 'cause of death', 'sunday:|:icon died': 'publication date', 'bearded boisterious:|:king tv': 'head of government', 'king tv:|:bearded boisterious': 'original network', 'bearded boisterious:|:pop culture': 'ethnic group', 'pop culture:|:bearded boisterious': 'genre', 'bearded boisterious:|:yell sell': 'performer', 'yell sell:|:bearded boisterious': 'shareholders', 'bearded boisterious:|:inlikely': 'mouth of the watercourse', 'inlikely:|:bearded boisterious': 'instance of', 'bearded boisterious:|:tampa fla': 'residence', 'tampa fla:|:bearded boisterious': 'residence', 'bearded boisterious:|:sunday': 'operator', 'sunday:|:bearded boisterious': 'operator', 'king tv:|:pop culture': 'official language', 'pop culture:|:king tv': 'genre', 'king tv:|:yell sell': 'replaced by', 'yell sell:|:king tv': 'replaced by', 'king tv:|:inlikely': 'followed by', 'inlikely:|:king tv': 'followed by', 'king tv:|:tampa fla': 'followed by', 'tampa fla:|:king tv': 'followed by', 'king tv:|:sunday': 'cast member', 'sunday:|:king tv': 'relative', 'pop culture:|:yell sell': 'record label', 'yell sell:|:pop culture': 'owned by', 'pop culture:|:inlikely': 'ethnic group', 'inlikely:|:pop culture': 'operator', 'pop culture:|:tampa fla': 'genre', 'tampa fla:|:pop culture': 'located in or next to body of water', 'pop culture:|:sunday': 'genre', 'sunday:|:pop culture': 'publication date', 'yell sell:|:inlikely': 'mouth of the watercourse', 'inlikely:|:yell sell': 'mouth of the watercourse', 'yell sell:|:tampa fla': 'owned by', 'tampa fla:|:yell sell': 'continent', 'yell sell:|:sunday': 'dissolved, abolished or demolished', 'sunday:|:yell sell': 'date of death', 'inlikely:|:tampa fla': 'followed by', 'tampa fla:|:inlikely': 'followed by', 'inlikely:|:sunday': 'point in time', 'sunday:|:inlikely': 'publication date', 'tampa fla:|:sunday': 'headquarters location', 'sunday:|:tampa fla': 'publication date'}
```

## Citing KeyBERT as the Primary Keyword Extraction Framework for KeyRelBERT:
[KeyBERT](https://github.com/MaartenGr/KeyBERT)
```bibtex
@misc{grootendorst2020keybert,
  author       = {Maarten Grootendorst},
  title        = {KeyBERT: Minimal keyword extraction with BERT.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.4461265},
  url          = {https://doi.org/10.5281/zenodo.4461265}
}
```
