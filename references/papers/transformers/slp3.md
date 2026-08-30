title: Speech and Language Processing
description: Speech and Language Processing

# Speech and Language Processing (3rd ed. draft) \
  [Dan Jurafsky](http://web.stanford.edu/people/jurafsky/) and [James H. Martin](http://www.cs.colorado.edu/~martin/)  

###  Here's our August 19, 2026 release! 

-  This release finally has a chapter 1! Much LLM material from the former chapter 7 was decanted here, plus lots of new material to make this a fun intro chapter for students as well as other newcomers to the field.

-  The former chapter 8 (Transformers) was merged with the remainder of former chapter 7 (\=pretraining and decoding/sampling) to create a new single chapter that covers transformers, pretraining, and decoding.

-  There is a large stub (about half) of an interpretability chapter 10, more to come on the next edition.

-  We used Claude Opus 5 to suggest more exercises for various chapters, and also to do a pass over the first 8 chapters to point out any bugs it could find. It found a lot. The result is some more exercises in this release, and also hopefully fewer typos and subtle notation inconsistencies to plague you the readers! 

 Individual chapters are below. **There are now updated slides for the intro and transformer/pretraining chapters!!**

 [Here is a single pdf of Aug 19, 2026 book!](https://web.stanford.edu/~jurafsky/slp3/ed3book_aug26.pdf) \

1.  **Feel free to use the draft chapters and slides in your classes**, print it out, whatever, the resulting feedback we get from you makes the book better! 
2.  **Typos and comments** are very welcome (just email [slp3edbugs@gmail.com](mailto:slp3edbugs@gmail.com) and let us know the date on the draft)! (Don't bother reporting missing refs due to cross-chapter cross-reference problems in the indvidual chapter pdfs, those are fixed in the full book draft)\

3.  **Gratitude!** We've put up a [list here](https://web.stanford.edu/~jurafsky/slp3/thanks.html) of the wonderful people who have sent so many fantastic suggestions and bug-fixes for improving the book. We are really grateful to all of you for your help, the book would not be possible without you! 
4. **How to cite the book**:
     Daniel Jurafsky and James H. Martin. 2026. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition with Language Models, 3rd edition. Online manuscript released August 19, 2026. https://web.stanford.edu/\~jurafsky/slp3. 

5.  A **bib entry** for the book is [**here**](https://web.stanford.edu/~jurafsky/slp3/jm3.bib).
    ```
    
    @Book{jm3,
      author =       "Daniel Jurafsky and James H. Martin",
      title =        "Speech and Language Processing: An Introduction to Natural Language Processing, 
      		  Computational Linguistics, and Speech Recognition,
    		   with Language Models",
      year =         "2026",
      url = {https://web.stanford.edu/~jurafsky/slp3/},
      note = "Online manuscript released August 19, 2026",
      edition =         "3rd",
      }
    ```
6.  **When** will the book be finished? Don't ask. 
7.  If you need the previous Jan 2026 draft chapters, they are [here](https://web.stanford.edu/~jurafsky/slp3/old_jan26/); If you need the previous Aug 2025 draft chapters, they are [here](https://web.stanford.edu/~jurafsky/slp3/old_aug25/); 

| **Volume I: Large Language Models** | ^^ | ^^ |
|:---|----|----|
|
|
|
|  | **Chapter** | **Slides**  |
| 1: |  [Introduction](https://web.stanford.edu/~jurafsky/slp3/1.pdf) |  1: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/intro_26aug.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/intro_26aug.pdf)\]<br>  |
| 2: |  [Words and Tokens](https://web.stanford.edu/~jurafsky/slp3/2.pdf) |  2: Words and Tokens \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/tokens_jan26.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/tokens_jan26.pdf)\] 2: Edit Distance \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/med24.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/med24.pdf)\]<br>  |
| 3: |  [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) |  3: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf)\]<br>  |
| 4: |  [ Logistic Regression and Text Classification](https://web.stanford.edu/~jurafsky/slp3/4.pdf) |  4: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/logreg25aug.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/logreg25aug.pdf)\]<br>  |
| 5: |  [ Embeddings](https://web.stanford.edu/~jurafsky/slp3/5.pdf) |  5: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/vector25aug.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/vector25aug.pdf)\]  |
| 6: |  [Neural Networks ](https://web.stanford.edu/~jurafsky/slp3/6.pdf)  |  6: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/nn25aug.pdf)\]  |
| 7: |  [Transformers and Pretraining](https://web.stanford.edu/~jurafsky/slp3/7.pdf)  | 7: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/transformer_aug26.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/transformer_aug26.pdf)\]  |
| 8: |  [Post-training](https://web.stanford.edu/~jurafsky/slp3/8.pdf) |
|   |
| **Volume II: Advanced LLM Topics and Tools** | ^^ | ^^ |
|  | **Chapter** | **Slides**  |
| 9: |  [Masked Language Models](https://web.stanford.edu/~jurafsky/slp3/9.pdf)  | 9: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/mlmjan25.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/mlmjan25.pdf)\]  |
| 10: |  [Interpretability](https://web.stanford.edu/~jurafsky/slp3/10.pdf)  |
| 11: |  [Information Retrieval and RAG](https://web.stanford.edu/~jurafsky/slp3/11.pdf)  | 11: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/ir_nov25.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/ir_nov25.pdf)\]  |
| 12: |  Agents \[not written yet\] |
| 13: |  [Machine Translation](https://web.stanford.edu/~jurafsky/slp3/13.pdf)  |
| 14: |  [RNNs and LSTMs](https://web.stanford.edu/~jurafsky/slp3/14.pdf)  | 13: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/rnnjan25.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/rnnjan25.pdf)\]  |
| 15: |  [Phonetics and Speech Feature Extraction](https://web.stanford.edu/~jurafsky/slp3/15.pdf) |
| 16: |  [Automatic Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/16.pdf) |
| 17: |  [Text-to-Speech](https://web.stanford.edu/~jurafsky/slp3/17.pdf) |
|   |
| **Volume III: Annotating Linguistic Structure** | ^^ | ^^ |
|  | **Chapter** | **Slides**  |
| 18: |  [Sequence Labeling for Parts of Speech and Named Entities](https://web.stanford.edu/~jurafsky/slp3/18.pdf) |  18: (Intro only) \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/8_POSNER_intro_May_6_2021.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/8_POSNER_intro_May_6_2021.pdf)\]  |
| 19: |  [Context-Free Grammars and Constituency Parsing](https://web.stanford.edu/~jurafsky/slp3/19.pdf) |
| 20: |  [Dependency Parsing ](https://web.stanford.edu/~jurafsky/slp3/20.pdf) |
| 21: |  [Information Extraction: Relations, Events, and Time](https://web.stanford.edu/~jurafsky/slp3/21.pdf) |
| 22: |  [Semantic Role Labeling and Argument Structure](https://web.stanford.edu/~jurafsky/slp3/22.pdf) |
| 23: |  [Lexicons for Sentiment, Affect, and Connotation ](https://web.stanford.edu/~jurafsky/slp3/23.pdf) |
| 24: |  [Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/24.pdf) |
| 25: |  [Discourse Coherence](https://web.stanford.edu/~jurafsky/slp3/25.pdf) |
| 26: |  [Conversation and its Structure](https://web.stanford.edu/~jurafsky/slp3/26.pdf) |
|   |
| **Appendix (will be just on the web)** | ^^ | ^^ |
| A: |  [Hidden Markov Models](https://web.stanford.edu/~jurafsky/slp3/A.pdf)  |
| B: |  [Naive Bayes Classification](https://web.stanford.edu/~jurafsky/slp3/B.pdf)  |  B: \[[pptx](https://web.stanford.edu/~jurafsky/slp3/slides/nb24aug.pptx)\] \[[pdf](https://web.stanford.edu/~jurafsky/slp3/slides/nb24aug.pdf)\]<br>  |
| C: |  [Kneser-Ney Smoothing](https://web.stanford.edu/~jurafsky/slp3/C.pdf)  |
| D: |  [Spelling Correction and the Noisy Channel](https://web.stanford.edu/~jurafsky/slp3/D.pdf)  |
| E: |  [Statistical Constituency Parsing](https://web.stanford.edu/~jurafsky/slp3/E.pdf)  |
| F: |  [Context-Free Grammars](https://web.stanford.edu/~jurafsky/slp3/F.pdf)  |
| G: |  [Combinatory Categorial Grammar](https://web.stanford.edu/~jurafsky/slp3/G.pdf)  |
| H: |  [Logical Representations of Sentence Meaning](https://web.stanford.edu/~jurafsky/slp3/H.pdf)  |
| I: |  [Word Senses and WordNet](https://web.stanford.edu/~jurafsky/slp3/I.pdf)  |
| J: |  [PPMI](https://web.stanford.edu/~jurafsky/slp3/J.pdf)  |
| K: |  [Frame-based Dialogue Systems](https://web.stanford.edu/~jurafsky/slp3/K.pdf)  |   |
