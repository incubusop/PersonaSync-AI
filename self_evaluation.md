[INFO] Loaded 140 samples across 5 classes.

[RESULT] Test accuracy: 0.643

[CLASSIFICATION REPORT]
                   precision    recall  f1-score   support

      action-item       1.00      0.67      0.80         6
emotional-support       0.80      0.67      0.73         6
         reminder       0.50      0.83      0.62         6
       small-talk       0.50      0.17      0.25         6
          unknown       0.57      1.00      0.73         4

         accuracy                           0.64        28
        macro avg       0.67      0.67      0.63        28
     weighted avg       0.68      0.64      0.62        28

[BENCHMARK] Avg inference latency: 0.43 ms/message (CPU)
[SIZE] classifier.pkl: 25.1 KB | vectorizer.pkl: 13.5 KB
[SIZE] Total: 0.04 MB (limit: 50 MB)