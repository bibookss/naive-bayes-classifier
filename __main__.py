import pandas as pd
from NaiveBayesClassifier import NaiveBayesSpamClassifier

if __name__ == '__main__':
    train_file = 'dataset/TrainingData.csv'
    test_file = 'dataset/TestData.csv'

    # Training 
    train_df = pd.read_csv(train_file, encoding='latin')
    X_train = train_df['message']
    y_train = train_df['label']    
    
    model = NaiveBayesSpamClassifier()
    model.fit(X_train, y_train)

    # Testing
    test_df = pd.read_csv(test_file, encoding='latin')
    X_test = test_df['message']
    # y_test = test_df['label']
    y_pred = model.predict(X_test)

    # Attach prediction to test data
    test_df['label'] = y_pred
    test_df.to_csv('dataset/PerezResultData.csv', index=False)
    print('Prediction saved to PerezResultData.csv')