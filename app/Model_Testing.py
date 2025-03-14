from tensorflow.keras.models import load_model

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

model = load_model('D:/Code/titleDefense/model/symptoms_checker_model.h5')
X_test = 10
y_test = 10

#model = load_model('../model/symptoms_checker_model.h5')

# Use some test data to check predictions
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=-1)

# Check the accuracy on the test data
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, predicted_classes))
