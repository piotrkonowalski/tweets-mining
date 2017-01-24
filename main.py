from collections import Counter

import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

stopwords = ["a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
             "before", "being", "between", "both", "by", "could", "did", "do", "does", "doing", "during", "each",
             "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
             "himself", "his", "how", "i", "in", "into", "is", "it", "its", "itself", "let", "me", "more", "most", "my",
             "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "own", "sha",
             "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
             "then", "there", "there's", "these", "they", "this", "those", "through", "to", "until", "up", "very",
             "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "with", "would", "you",
             "your", "yours", "yourself", "yourselves",
             "n't", "'s", "'ll", "'re", "'d", "'m", "'ve",
             "didn't", "he's", "she's", "i'll", "you'll", "she'll", "he'll", "we'll", "they'll", "you're", "we're",
             "i'd", "you'd", "i'm", "i've", "you've", "we've",
             "above", "again", "against", "below", "but", "cannot", "down", "few", "if", "off",
             "out", "over", "same", "too", "under", "why",
             # "no", "nor", "not",
             ]

punctuationChars = [',', '.', ':', '?', '!', '*', '(', ')', '"', "'", ';', '(', ')', '...', '-', '&', '^', '*', '%',
                    '$', '#']
stopwords += punctuationChars

ps = PorterStemmer()
lmt = WordNetLemmatizer()
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

tweets = pd.read_csv('train.csv', '.', ',', header=0)


def prepare_tokens(text):
    tokens = tokenizer.tokenize(text)
    tokens = list(filter(lambda t: t not in stopwords, tokens))
    tokens = [ps.stem(lmt.lemmatize(t)) for t in tokens]

    return tokens


def get_common_words():
    words = Counter()
    for i in tweets.index:
        tweet = tweets.loc[i, 'Tweet']
        tokens = prepare_tokens(tweet)
        words.update(tokens)

    return words


def create_bow(documents, features, read_labels=True):
    row = []
    col = []
    data = []

    labels = []

    for i in documents.index:
        tweet = documents.loc[i, 'Tweet']
        tweet_tokens = prepare_tokens(tweet)

        if read_labels:
            label = documents.loc[i, 'Category']
            labels.append(label)

        for token in set(tweet_tokens):
            if token not in features:
                continue
            row.append(i)
            col.append(features[token])
            data.append(1)
    return csr_matrix((data, (row, col)), shape=(len(documents), len(features))), labels


def test_classifier(classifier, dataset, feature_dict, list_of_labels):
    x_test, _ = create_bow(dataset, feature_dict)
    predicted = classifier.predict(x_test)
    expected = dataset.loc[:, 'Category']
    print_results(expected, predicted, list_of_labels)


def print_results(expected, predicted, list_of_labels):
    print("=================== Results ===================")
    print("            Positive    Neutral     Negative   ")
    print("F1       ", f1_score(expected, predicted, average=None, pos_label=None, labels=list_of_labels))
    print("Precision", precision_score(expected, predicted, average=None, pos_label=None, labels=list_of_labels))
    print("Recall   ", recall_score(expected, predicted, average=None, pos_label=None, labels=list_of_labels))


def main():
    common_words = get_common_words()
    common_words = list([k for k, v in common_words.most_common() if v > 5])

    feature_dict = {}
    for word in common_words:
        feature_dict[word] = len(feature_dict)

    train, test = train_test_split(tweets, test_size=0.25, random_state=42)
    columns = train.columns
    train = pd.DataFrame(train.as_matrix())
    train.columns = columns
    test = pd.DataFrame(test.as_matrix())
    test.columns = columns

    print("Training classifier...")

    dataset = tweets
    # dataset = train

    x_train, y_train = create_bow(dataset, feature_dict)
    list_of_labels = list(set(y_train))

    forrest = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=23)
    forrest.fit(x_train, y_train)

    # ada = AdaBoostClassifier(n_estimators=300, )
    # ada.fit(x_train, y_train)
    #
    # nb = MultinomialNB()
    # nb.fit(x_train, y_train)
    #
    # knn = KNeighborsClassifier()
    # knn.fit(x_train, y_train)

    # classifiers = [forrest, ada, nb, knn]

    classifiers = [forrest]

    print("Testing...")

    # for classifier in classifiers:
    #     test_classifier(classifier, test, feature_dict, list_of_labels)

    # test_classifier(forrest, test, feature_dict, list_of_labels)

    test = pd.read_csv("test.csv", '.', ',', header=0)
    x_test, _ = create_bow(test, feature_dict, read_labels=False)
    predicted = forrest.predict(x_test)
    test['Category'] = pd.Series(predicted, index=test.index)

    # for i in test.head().index:
    #     tokens = prepare_tokens(test.loc[i, 'Tweet'])
    #     if not set([':)', ':-)', ';)', ';-)']).isdisjoint(tokens):
    #         test.loc[i, 'Category'] = 'positive'
    #
    #     if not set([':(', ':-(', ';(', ';-(']).isdisjoint(tokens):
    #         test.loc[i, 'Category'] = 'negative'

    result = test.loc[:, ['Id', 'Category']]

    result.to_csv('result.csv', index=False)


if __name__ == "__main__":
    main()
