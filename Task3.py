import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - using modern .keras format")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - using scikit-learn")

class HandwrittenDigitRecognition:
    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        
    def load_data(self):
        print("Loading MNIST dataset...")
        try:
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            X, y = mnist.data, mnist.target.astype(int)
            X = X[:10000]
            y = y[:10000]
            print(f"Data shape: {X.shape}")
            return X, y
        except Exception as e:
            print(f"Error: {e}")
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_samples=5000):
        print("Generating synthetic data...")
        X = []
        y = []
        
        for i in range(n_samples):
            digit = i % 10
            img = np.zeros(784)
            
            if digit == 0:
                img[100:150] = 1
                img[600:650] = 1
                img[100:600:28] = 1
                img[149:649:28] = 1
            elif digit == 1:
                img[120:620:28] = 1
            elif digit == 2:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[149:300:28] = 1
            elif digit == 3:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[149:649:28] = 1
            elif digit == 4:
                img[100:300:28] = 1
                img[300:350] = 1
                img[149:649:28] = 1
            elif digit == 5:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[100:300:28] = 0.5
                img[149:649:28] = 1
            elif digit == 6:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[100:649:28] = 1
            elif digit == 7:
                img[100:150] = 1
                img[149:649:28] = 1
            elif digit == 8:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[100:649:28] = 1
                img[149:649:28] = 1
            elif digit == 9:
                img[100:150] = 1
                img[300:350] = 1
                img[600:650] = 1
                img[149:649:28] = 1
                img[100:300:28] = 0.5
            
            img += np.random.normal(0, 0.1, 784)
            img = np.clip(img, 0, 1)
            
            X.append(img)
            y.append(digit)
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, X, y):
        X = X / 255.0
        return X, y
    
    def create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def train_sklearn_models(self, X_train, y_train, X_test, y_test):
        print("Training scikit-learn models...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42),
            'SVM (Linear)': SVC(kernel='linear', random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            y_pred = model.predict(X_test)
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'predictions': y_pred
            }
            print(f"{name} - Train: {train_score:.4f}, Test: {test_score:.4f}")
        
        return results
    
    def train_cnn_model(self, X_train, y_train, X_test, y_test):
        print("Training CNN model...")
        X_train_cnn = X_train.reshape(-1, 28, 28, 1)
        X_test_cnn = X_test.reshape(-1, 28, 28, 1)
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        model = self.create_cnn_model()
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        
        history = model.fit(
            X_train_cnn, y_train_cat,
            batch_size=128,
            epochs=20,
            validation_data=(X_test_cnn, y_test_cat),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        loss, accuracy = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        y_pred_proba = model.predict(X_test_cnn)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        return {
            'model': model,
            'train_score': history.history['accuracy'][-1],
            'test_score': accuracy,
            'predictions': y_pred,
            'history': history
        }
    
    def evaluate_models(self, results, y_test, model_type='sklearn'):
        best_model_name = None
        best_score = 0
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"Training Accuracy: {result['train_score']:.4f}")
            print(f"Test Accuracy: {result['test_score']:.4f}")
            
            if result['test_score'] > best_score:
                best_score = result['test_score']
                best_model_name = name
        
        print(f"\nBest model: {best_model_name} (Accuracy: {best_score:.4f})")
        
        best_result = results[best_model_name]
        cm = confusion_matrix(y_test, best_result['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        print(f"\nClassification Report for {best_model_name}:")
        print(classification_report(y_test, best_result['predictions']))
        
        if model_type == 'cnn' and 'history' in best_result:
            history = best_result['history']
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return best_model_name, best_result['model']
    
    def visualize_predictions(self, model, X_test, y_test, num_images=12, model_type='sklearn'):
        if model_type == 'cnn':
            X_test_cnn = X_test.reshape(-1, 28, 28, 1)
            y_pred_proba = model.predict(X_test_cnn)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        plt.figure(figsize=(15, 10))
        for i in range(num_images):
            idx = np.random.randint(0, X_test.shape[0])
            plt.subplot(3, 4, i+1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_test[idx]}, Pred: {y_pred[idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    print("Handwritten Digit Recognition")
    print("=" * 50)
    
    hdr = HandwrittenDigitRecognition()
    X, y = hdr.load_data()
    X, y = hdr.preprocess_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if TENSORFLOW_AVAILABLE:
        cnn_result = hdr.train_cnn_model(X_train, y_train, X_test, y_test)
        results = {'CNN': cnn_result}
        best_model_name, best_model = hdr.evaluate_models(results, y_test, model_type='cnn')
        hdr.visualize_predictions(best_model, X_test, y_test, model_type='cnn')
        best_model.save('cnn_digit_recognition.keras')
        print("CNN model saved as 'cnn_digit_recognition.keras'")
    else:
        results = hdr.train_sklearn_models(X_train, y_train, X_test, y_test)
        best_model_name, best_model = hdr.evaluate_models(results, y_test)
        hdr.visualize_predictions(best_model, X_test, y_test)
        import joblib
        joblib.dump(best_model, 'sklearn_digit_model.pkl')
        print(f"{best_model_name} model saved as 'sklearn_digit_model.pkl'")

if __name__ == "__main__":
    main()