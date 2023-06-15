import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Suppose you have your text corpus as a list of sentences
corpus = ["ABABABABA", "CDCDCDCD", "EFEFEFEF"]

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
for i, context in enumerate(contexts):
    for ascii_val in context:
        X[i, ascii_val] += 1

# Apply the weights
weights = [1 / (i + 1) for i in range(1, X.shape[1] + 1)]
X = X * weights

# Encode the target characters as integers
y = [int(char) for char in targets]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the multiclass SVM
model = svm.SVC(decision_function_shape="ovo")
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)


def decode(model, initial_context, length):
    text = initial_context
    for _ in range(length):
        # Transform the current context into the binary matrix representation
        X = np.zeros((1, 256))
        context = [ord(char) for char in text[-20:]]
        for ascii_val in context:
            X[0, ascii_val] += 1
        X = X * weights
        # Predict the next character
        prediction = model.predict(X)
        # Decode the prediction
        next_char = chr(int(prediction))
        # Add the predicted character to the text
        text += next_char
    return text


# Use the decoder to generate text
generated_text = decode(model, "AB", 30)
print(generated_text)
