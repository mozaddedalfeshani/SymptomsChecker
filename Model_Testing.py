from tensorflow.keras.models import load_model
model = load_model('symptoms_checker_model.h5')

# Use some test data to check predictions
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=-1)

# Check the accuracy on the test data
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, predicted_classes))
