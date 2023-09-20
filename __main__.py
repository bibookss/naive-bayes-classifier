import pandas as pd
from models.NaiveBayesClassifier import NaiveBayesSpamClassifier
from metrics.precision import precision
from metrics.recall import recall

if __name__ == '__main__':
    train_file = 'dataset/TrainingData.csv'
    test_file = 'dataset/TestData.csv'
    valid_file = 'dataset/LabeledTestData.csv'

    # Training 
    train_df = pd.read_csv(train_file, encoding='latin')
    X_train = train_df['message']
    y_train = train_df['label']    
    
    model = NaiveBayesSpamClassifier()
    model.fit(X_train, y_train)

    # Testing
    test_df = pd.read_csv(test_file, encoding='latin')
    X_test = test_df['message']
    y_pred = model.predict(X_test)

    # Evaluation
    valid_df = pd.read_csv(valid_file, encoding='latin')
    y_valid = valid_df['label']

    # print(len(y_valid) == len(y_pred))

    print('Precision: {}'.format(precision(y_valid, y_pred)))
    print('Recall: {}'.format(recall(y_valid, y_pred)))

    # Attach prediction to test data
    test_df.insert(0, 'label', y_pred)
    test_df.to_csv('dataset/PerezResultData.csv', index=False)
    print('Prediction saved to PerezResultData.csv')