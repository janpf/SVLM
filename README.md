# SVLM: Support Vector Language Model

Character-based Autoregressive Language Modelling only using Support Vector Machines (SVMs).

## Key Techniques

- Character-wise tokenization: The text is tokenized at the character level, with each character represented by its ASCII value. This results in numerical tokens that can be used as inputs to the SVM. (basically ByT5 tokenization)

- Context generation: For each character, a context window of the previous N characters is created. These context windows serve as the input sequences for the SVM. (basically GPT)

- Token Embeddings: Each context is represented as a matrix, with each column corresponding to a timestep and each entry in a context's row represents the weighted presence of a character in that context. The characters are indexed by their ASCII value.

- Token Weighting scheme: A harmonic series is used to weight the presence of characters in a context, with more recent characters having a higher weight.

- Autoregressive prediction: The trained SVM model is used to predict the next character given a context. This prediction is then added to the context, and the process is repeated to generate new text.
