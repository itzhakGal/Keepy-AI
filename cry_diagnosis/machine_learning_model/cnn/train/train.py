from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

# Load features and labels
with open('../model/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('../model/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

print(f"Loaded {len(features)} features and {len(labels)} labels")

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
input_shape = (X_train.shape[1], 1)
num_classes = y_train.shape[1]

# Create model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = create_model(input_shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('baby_cry_cnn_model.h5')
