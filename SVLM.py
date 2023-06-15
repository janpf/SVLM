import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

max_seq_len = 30

# Suppose you have your text corpus as a list of sentences
corpus = [
    "ABABABABA",
    "CDCDCDCD",
    "EFEFEFEF",
    "AAABBBCCCDDD",
    "ABCDEABCDEABCDE",
    "AAAABBBBAAAABBBB",
    "ABCDABCDAB",
    "ABCCBAABCCBA",
]

# Tokenize the corpus characterwise by their ASCII code points
tokens = [[ord(char) for char in sentence] for sentence in corpus]

# Create input subsequences and corresponding next character for every single subsequence of tokens
contexts, targets = [], []
for sequence in tokens:
    for i in range(1, len(sequence)):
        # Consider only the last 20 characters
        contexts.append(sequence[max(i - 20, 0) : i])
        targets.append(sequence[i])

# Transform contexts to a binary matrix representation
X = np.zeros((len(contexts), 256))
# with weights, so that the last character has the highest weight
weights = [1 / (i + 1) for i in range(max_seq_len)]
for i, context in enumerate(contexts):
    for j, ascii_val in enumerate(reversed(context)):
        X[i, ascii_val] += weights[j]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

# Create and train the multiclass SVM
model = svm.SVC(decision_function_shape="ovo")
model.fit(X_train, y_train)


def decode(model, initial_context, length):
    text = initial_context
    for _ in range(length):
        # Transform the current context into the binary matrix representation
        X = np.zeros((1, 256))
        context = [ord(char) for char in text[-20:]]
        for i, ascii_val in enumerate(reversed(context)):
            X[0, ascii_val] += weights[i]
        # Predict the next character
        prediction = model.predict(X)
        # Decode the prediction
        next_char = chr(int(prediction))
        # Add the predicted character to the text
        text += next_char
    return text


# Use the decoder to generate text
generated_text = decode(model, "A", max_seq_len)
print(generated_text)
