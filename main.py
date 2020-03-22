import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from dataset import WrapperDataset
from constants import *
from training import train, evaluate
from models import Network

# Load the data
if __name__ == '__main__':
    df = pd.read_csv('data\\news.csv')
    print('Shape of the data: {}'.format(df.shape))

    # Drop Unnamed:0
    df = df.drop('Unnamed: 0', axis=1)

    # Extract the label
    labels = df.label

    # Train and test splitting
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=1)
    # Train and validation splitting
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    # Let's plot the distribution of the classes
    plt.figure(figsize=(12, 5))
    plt.hist(y_train)
    plt.show()

    # Initialize a TF-IDF vectorizer. Stop words are taken from English vocabulary,
    # with a maximum term frequency MAX_TERM_FREQUENCY,
    # i.e. term with a frequency larger than MAX_TERM_FREQUENCY will be discarded
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=MAX_TERM_FREQUENCY)

    # Let's fit it on train and test
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_val = tfidf_vectorizer.transform(x_val)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    # Create a dataframe out of tfidf_train
    df_idf = pd.DataFrame(tfidf_vectorizer.idf_, index=tfidf_vectorizer.get_feature_names(), columns=['idf_weights'])
    df_idf.reset_index(inplace=True)
    df_idf.rename(columns={'index': 'word'}, inplace=True)

    # Let's see the words with the highest TF-IDF
    sorted_idf = df_idf.sort_values(by='idf_weights')
    sorted_idf_top_10 = sorted_idf.head(10)

    plt.figure(figsize=(12, 5))
    plt.hist(sorted_idf_top_10.word, weights=sorted_idf_top_10.idf_weights)
    # plt.show()

    # Let's generate the dataset and the loaders
    train_dataset = WrapperDataset(tfidf_train.toarray(), y_train)
    valid_dataset = WrapperDataset(tfidf_val.toarray(), y_val)
    test_dataset = WrapperDataset(tfidf_test.toarray(), y_test)

    train_dl = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Init the network
    net = Network(input=tfidf_train.shape[1], layers=3, hidden=1000, output=2)

    # Training
    net = train(net, train_dl, valid_dl, epochs=500, early_stopping=False)

    # Test
    test_accuracy, test_loss = evaluate(net, test_dl, torch.nn.CrossEntropyLoss())

    print('Final test accuracy: {:.2f}'.format(test_accuracy))
