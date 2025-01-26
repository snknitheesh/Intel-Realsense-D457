# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.signal import butter, lfilter

# 1. Data Simulation (Replace this with actual sensor data)
def simulate_data(num_samples=1000):
    """Simulates eyelid sensor data with noise."""
    np.random.seed(42)
    time = np.linspace(0, 10, num_samples)
    # Simulate "open" signal as a sine wave
    open_signal = np.sin(2 * np.pi * 1 * time) + np.random.normal(0, 0.2, num_samples)
    # Simulate "closed" signal as a flat signal with noise
    closed_signal = np.random.normal(0, 0.2, num_samples)
    # Concatenate and create labels
    data = np.concatenate([open_signal, closed_signal])
    labels = np.array([0] * num_samples + [1] * num_samples)  # 0: Open, 1: Closed
    return data, labels

# Simulated data
data, labels = simulate_data()
plt.plot(data[:1000], label='Open')
plt.plot(data[1000:], label='Closed')
plt.legend()
plt.title("Simulated Eyelid Movement Data")
plt.show()

# 2. Preprocessing: Apply a Butterworth Filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Filter Parameters
cutoff_frequency = 2.0  # Hz
sampling_rate = 50.0  # Hz
filtered_data = butter_lowpass_filter(data, cutoff_frequency, sampling_rate)

plt.plot(filtered_data[:1000], label='Filtered Open')
plt.plot(filtered_data[1000:], label='Filtered Closed')
plt.legend()
plt.title("Filtered Data")
plt.show()

# 3. Feature Extraction
def extract_features(data, window_size=50):
    """Extracts statistical features from sliding windows of data."""
    features = []
    for start in range(0, len(data) - window_size, window_size):
        window = data[start:start + window_size]
        features.append([np.mean(window), np.std(window), np.min(window), np.max(window)])
    return np.array(features)

window_size = 50
features = extract_features(filtered_data, window_size)
labels_downsampled = labels[:len(features) * window_size:window_size]  # Downsample labels

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels_downsampled, test_size=0.2, random_state=42)

# 5. Model Training: Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Real-Time Prediction Simulation
def simulate_real_time(data, model, window_size):
    """Simulates real-time prediction from sliding window data."""
    for start in range(0, len(data) - window_size, window_size):
        window = data[start:start + window_size]
        features = [np.mean(window), np.std(window), np.min(window), np.max(window)]
        prediction = model.predict([features])[0]
        print(f"Window {start}-{start + window_size}: Predicted Label = {'Closed' if prediction else 'Open'}")

simulate_real_time(filtered_data, clf, window_size)
