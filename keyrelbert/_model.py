import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from typing import List, Union, Tuple
import string

from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from keyrelbert._mmr import mmr
from keyrelbert._maxsum import max_sum_distance
from keyrelbert._highlight import highlight_document
from keyrelbert.backend._utils import select_backend
import torch
from tqdm import tqdm

class KeyRelBERT:
    """
    A minimal method for keyword extraction with BERT

    The keyword extraction is done by finding the sub-phrases in
    a document that are the most similar to the document itself.

    First, document embeddings are extracted with BERT to get a
    document-level representation. Then, word embeddings are extracted
    for N-gram words/phrases. Finally, we use cosine similarity to find the
    words/phrases that are the most similar to the document.

    The most similar words could then be identified as the words that
    best describe the entire document.

    <div class="excalidraw">
    --8<-- "docs/images/pipeline.svg"
    </div>
    """

    def __init__(self, model="all-MiniLM-L6-v2"):
        """KeyRelbert initialization

        Arguments:
            model: Use a custom embedding model.
                   The following backends are currently supported:
                      * SentenceTransformers
                      * 🤗 Transformers
                      * Flair
                      * Spacy
                      * Gensim
                      * USE (TF-Hub)
                    You can also pass in a string that points to one of the following
                    sentence-transformers models:
                      * https://www.sbert.net/docs/pretrained_models.html
        """
        self.model = select_backend(model)
        self.criterion = torch.nn.CosineEmbeddingLoss()
        self.relations = ['located in the administrative territorial entity', 'parent organization', 'publication date', 'religion', 'alternate name', 'continent', 'instance of', 'replaced by', 'official language', 'narrative location', 'killed by', 'genre', 'charge', 'ethnicity', 'occupation', 'basin country', 'work location', 'denonym', 'member of sports team', 'participant', 'languages spoken, written or signed', 'head of state', 'production company', 'position held', 'member count', 'developer', 'spouse', 'product or material produced', 'founded by', 'operator', 'sibling', 'notable work', 'major shareholder', 'age', 'state of headquarters', 'participant of', 'unemployment rate', 'original language of work', 'present in work', 'award received', 'dissolved, abolished or demolished', 'country of headquarters', 'educated at', 'head of government', 'producer', 'series', 'applies to jurisdiction', 'residence', 'military branch', 'country of citizenship', 'publisher', 'country of origin', 'advisors', 'part of', 'identity', 'location of formation', 'located on terrain feature', 'employer', 'has member', 'father', 'inception', 'member of political party', 'state of birth', 'country of residence', 'relative', 'country of death', 'member of', 'start time', 'creator', 'mouth of the watercourse', 'composer', 'league', 'follows', 'country', 'website', 'top members', 'city of residence', 'territory claimed by', 'contains administrative territorial entity', 'owned by', 'followed by', 'subclass of', 'headquarters location', 'dissolved', 'ethnic group', 'neighborhood of', 'chairperson', 'platform', 'located in or next to body of water', 'date of birth', 'manufacturer', 'replaces', 'lyrics by', 'state of death', 'author', 'shareholders', 'country of birth', 'sister city', 'date of death', 'subsidiary', 'director', 'state of residence', 'record label', 'influenced by', 'capital of', 'performer', 'mother', 'has part', 'place of birth', 'original network', 'parent taxon', 'conflict', 'place of death', 'capital', 'screenwriter', 'end time', 'separated from', 'cause of death', 'affiliation', 'child', 'legislative body', 'no relation', 'point in time', 'characters', 'cast member', 'location', 'industry']
        self.relation_embeddings = self.get_relation_embeddings()

    def get_relation_embeddings(self):
        return self.model.embed(self.relations)
        
    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: CountVectorizer = None,
        highlight: bool = False,
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract keywords and/or keyphrases

        To get the biggest speed-up, make sure to pass multiple documents
        at once instead of iterating over a single document.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.
                           NOTE: when multiple documents are passed, 
                           `seed_keywords`funtions in either of the two ways below:
                           - globally: when a flat list of str is passed, keywords are shared by all documents, 
                           - locally: when a nested list of str is passed, keywords differs among documents.
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.

        Returns:
            keywords: The top n keywords for a document with their respective distances
                      to the input document.

        Usage:

        To extract keywords from a single document:

        ```python
        from keyrelbert import KeyRelBert

        kw_model = KeyRelBert()
        keywords = kw_model.extract_keywords(doc)
        ```

        To extract keywords from multiple documents, which is typically quite a bit faster:

        ```python
        from keyrelbert import KeyRelBert

        kw_model = KeyRelBert()
        keywords = kw_model.extract_keywords(docs)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(docs)

        # Check if the right number of word embeddings are generated compared with the vectorizer
        if word_embeddings is not None:
            if word_embeddings.shape[0] != len(words):
                raise ValueError("Make sure that the `word_embeddings` are generated from the function "
                                 "`.extract_embeddings`. \nMoreover, the `candidates`, `keyphrase_ngram_range`,"
                                 "`stop_words`, and `min_df` parameters need to have the same values in both "
                                 "`.extract_embeddings` and `.extract_keywords`.")

        # Extract embeddings
        if doc_embeddings is None:
            doc_embeddings = self.model.embed(docs)
        if word_embeddings is None:
            word_embeddings = self.model.embed(words)
        doc_embeddings = doc_embeddings.numpy()
        word_embeddings = word_embeddings.numpy()

        # Guided KeyRelBert either local (keywords shared among documents) or global (keywords per document)
        if seed_keywords is not None:
            if isinstance(seed_keywords[0], str):
                seed_embeddings = self.model.embed(seed_keywords).mean(axis=0, keepdims=True)    
            elif len(docs) != len(seed_keywords):
                raise ValueError("The length of docs must match the length of seed_keywords")
            else:
                seed_embeddings = np.vstack([
                    self.model.embed(keywords).mean(axis=0, keepdims=True)
                    for keywords in seed_keywords
                ])
            doc_embeddings = ((doc_embeddings * 3 + seed_embeddings) / 4)

        # Find keywords
        all_keywords = []
        for index, _ in enumerate(docs):

            try:
                # Select embeddings
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

                # Maximal Marginal Relevance (MMR)
                if use_mmr:
                    keywords = mmr(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        diversity,
                    )

                # Max Sum Distance
                elif use_maxsum:
                    keywords = max_sum_distance(
                        doc_embedding,
                        candidate_embeddings,
                        candidates,
                        top_n,
                        nr_candidates,
                    )

                # Cosine-based keyword extraction
                else:
                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    keywords = [
                        (candidates[index], round(float(distances[0][index]), 4))
                        for index in distances.argsort()[0][-top_n:]
                    ][::-1]

                all_keywords.append(keywords)

            # Capturing empty keywords
            except ValueError:
                all_keywords.append([])

        # Highlight keywords in the document
        if len(all_keywords) == 1:
            if highlight:
                highlight_document(docs[0], all_keywords[0], count)
            all_keywords = all_keywords[0]

        return all_keywords

    def extract_embeddings(
        self,
        docs: Union[str, List[str]],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        min_df: int = 1,
        vectorizer: CountVectorizer = None
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Extract document and word embeddings for the input documents and the
        generated candidate keywords/keyphrases respectively.

        Note that all potential keywords/keyphrases are not returned but only their
        word embeddings. This means that the values of `candidates`, `keyphrase_ngram_range`,
        `stop_words`, and `min_df` need to be the same between using `.extract_embeddings` and
        `.extract_keywords`.

        Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`

        Returns:
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.

        Usage:

        To generate the word and document embeddings from a set of documents:

        ```python
        from keyrelbert import KeyRelBert

        kw_model = KeyRelBert()
        doc_embeddings, word_embeddings = kw_model.extract_embeddings(docs)
        ```

        You can then use these embeddings and pass them to `.extract_keywords` to speed up the tuning the model:

        ```python
        keywords = kw_model.extract_keywords(docs, doc_embeddings=doc_embeddings, word_embeddings=word_embeddings)
        ```
        """
        # Check for a single, empty document
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []

        # Extract potential words using a vectorizer / tokenizer
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        word_embeddings = self.model.embed(words)
        doc_embeddings = self.model.embed(docs)

        return doc_embeddings, word_embeddings

    def backpropogate_and_get_words(self, docs, w_embedding):
        self.model.embedding_model.zero_grad()
        word_embedding_layer = self.model.embedding_model.get_submodule('0.auto_model.embeddings.word_embeddings')
        output = self.model.embed(docs, return_detached=True)[0]
        single_word_embeddings = output['sentence_embedding']
        input_ids = output['input_ids']
        loss = self.criterion(single_word_embeddings, w_embedding, torch.tensor(1))
        loss.backward()
        gradients = word_embedding_layer.weight.grad
        gradients = torch.index_select(gradients, 0, input_ids)
        gradients_columnar = torch.mean(gradients, 1), torch.std(gradients, 1)
        gradients = gradients.transpose(0, 1)
        gradients = (gradients - gradients_columnar[0])/gradients_columnar[1]
        gradients = gradients.transpose(0, 1)
        scores = torch.sum(torch.abs(gradients), 1).numpy()
        input_ids = input_ids.numpy()
        sorted_indices = np.argsort(scores)[::-1]
        ranked_scores = scores[sorted_indices]
        thr_index = len([i for i in ranked_scores if i>ranked_scores[0]-np.std(scores)/1.5])
        sorted_indices = sorted_indices[:thr_index]
        ranked_ids = input_ids[sorted_indices]
        ranked_ids_ = []
        for r in ranked_ids:
            if r not in ranked_ids_:
                ranked_ids_.append(r)
        return ranked_ids_

    def extract_relations(
        self,
        docs: Union[str, List[str]],
        keywords: Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]],
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
        if_use_rank_ids = True,
        verbose=True
    ):
        if isinstance(docs, str):
            if docs:
                docs = [docs]
            else:
                return []
        if vectorizer:
            count = vectorizer.fit(docs)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(docs)
            except ValueError:
                return []
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        words = words.tolist()
        if if_use_rank_ids:
            doc_embeddings = doc_embeddings.repeat(self.relation_embeddings.shape[0], 1)
        relations = {}
        for idx_a in tqdm(range(len(keywords)), desc="extracting_relations", disable=not verbose):
            word_a = keywords[idx_a][0]
            w_embedding_a = word_embeddings[words.index(word_a)]
            for idx_b in range(len(keywords))[idx_a+1:]:
                word_b = keywords[idx_b][0]
                w_embedding_b = word_embeddings[words.index(word_b)]
                relation_sent = (w_embedding_a.repeat(self.relation_embeddings.shape[0], 1)*1+self.relation_embeddings+w_embedding_b.repeat(self.relation_embeddings.shape[0], 1)*1)/(1+2*1)
                if if_use_rank_ids:
                    ranked_ids = self.backpropogate_and_get_words(docs, (w_embedding_a+w_embedding_b)/2)
                    competent_words = " ".join([w for w in self.model.embedding_model.tokenizer.convert_ids_to_tokens(ranked_ids) if w not in string.punctuation])
                    subset_a = word_a+" "+competent_words+" "+word_b
                    subset_b = word_b+" "+competent_words+" "+word_b
                    subset_embeddings = self.model.embed([subset_a, subset_b])
                    subset_a, subset_b = subset_embeddings
                    subset_a = subset_a.repeat(self.relation_embeddings.shape[0], 1)
                    subset_b = subset_b.repeat(self.relation_embeddings.shape[0], 1)
                    similarities_a = torch.nn.functional.cosine_similarity(subset_a, relation_sent, dim=1).numpy()
                    similarities_b = torch.nn.functional.cosine_similarity(subset_b, relation_sent, dim=1).numpy()
                    relation_a = self.relations[np.argmax(similarities_a)]
                    relation_b = self.relations[np.argmax(similarities_b)]
                else:
                    similarities = torch.nn.functional.cosine_similarity(doc_embeddings, relation_sent, dim=1).numpy()
                    relation_a = self.relations[np.argmax(similarities)]
                    relation_b = relation_a
                relations[word_a+":|:"+word_b] = relation_a
                relations[word_b+":|:"+word_a] = relation_b
        return relations