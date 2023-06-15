import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# suppose you have your text corpus as a list of sentences
corpus = ["ababababab"] * 100  # big data

# tokenize the corpus characterwise by their ASCII code points
tokens = [[ord(char) for char in sentence] for sentence in corpus]

# create input subsequences and corresponding next character for every single subsequence of tokens
contexts, targets = [], []

for sequence in tokens:
    for i in range(1, len(sequence)):
        # Consider only the last 20 characters
        contexts.append(sequence[max(i - 20, 0) : i])
        targets.append(sequence[i])

## flatten the list of contexts to use with CountVectorizer
contexts = [" ".join(map(str, context)) for context in contexts]


# custom transformer that applies the weights
class CustomTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        weights = [1 / (i + 1) for i in range(X.shape[1], 0, -1)]
        matrix = np.tril(np.ones((X.shape[1], X.shape[1])) * weights)
        return X.dot(matrix)


# create a pipeline that uses CountVectorizer and applies the weights
pipeline = make_pipeline(
    CountVectorizer(tokenizer=lambda x: x.split(" "), dtype=np.float64, ngram_range=(1, 2)),
    # CustomTransformer(),
)
# transform the contexts
X = pipeline.fit_transform(contexts)
# encode the target characters as integers
y = [int(char) for char in targets]

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train the multiclass SVM
model = svm.SVC(
    decision_function_shape="ovr",
    kernel="linear",
)
model.fit(X_train, y_train)

# make predictions on the test set
# predictions = model.predict(X_test)


def decode(model, pipeline, initial_context, length):
    text = initial_context
    for _ in range(length):
        # Vectorize the current context
        X = pipeline.transform([text])
        # Predict the next character
        prediction = model.predict(X)
        # Decode the prediction
        next_char = chr(int(prediction))

        # Add the predicted character to the text
        text += next_char
    return text


# Use the decoder to generate text
generated_text = decode(
    model,
    pipeline,
    "a",
    50,
)
print(generated_text)
