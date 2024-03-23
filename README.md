# SVLM: Support Vector Language Model

Character-based Autoregressive Language Modelling only using Support Vector Machines (SVMs).

## Key Techniques

- Character-wise tokenization: The text is tokenized at the character level, with each character represented by its ASCII value. This results in numerical tokens that can be used as inputs to the SVM.

- Context generation: For each character, a context window of the previous N characters is created. These context windows serve as the input sequences for the SVM. (basically GPT)

- Token Embeddings: The context to generate a new token is represented as a vector, where each entry corresponds to a character from the vocabulary. The characters are indexed by their ASCII value.

- Token Weighting scheme: A harmonic series is used to weight the presence of characters in a context, with more recent characters having a higher weight (higher value in the vector).

- Autoregressive prediction: The trained SVM model is used to predict the next character given a context. This prediction is then added to the context, and the process is repeated to generate new text.

## Why?

We published a RandomForest-based version (clearly superior over SVLM) of this in https://ceur-ws.org/Vol-3630/LWDA2023-paper33.pdf.
