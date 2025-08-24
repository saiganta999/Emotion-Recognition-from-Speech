import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class EmotionRecognition:
    def __init__(self):
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        self.features = []
        self.labels = []
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        
    def generate_synthetic_audio_features(self, n_samples=1000):
        print("Generating synthetic audio emotion data...")
        
        for i in range(n_samples):
            emotion = np.random.choice(self.emotions)
            
            features = np.random.normal(0, 1, 193)
            
            emotion_idx = self.emotions.index(emotion)
            features = features + (emotion_idx * 0.2)
            
            self.features.append(features)
            self.labels.append(emotion)
        
        print(f"Generated {len(self.features)} synthetic audio samples")
        
        X = np.array(self.features)
        y = self.le.fit_transform(self.labels)
        
        return X, y

    def create_model(self):
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced'),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        return models

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        models = self.create_model()
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
            
            print(f"{name} - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        return results, X_test, y_test

    def evaluate(self, results, y_test):
        print("Evaluating models...")
        
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
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.le.classes_, yticklabels=self.le.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        print(f"\nClassification Report for {best_model_name}:")
        print(classification_report(y_test, best_result['predictions'], 
                                  target_names=self.le.classes_))
        
        return best_model_name, results[best_model_name]['model']

    def visualize_predictions(self, model, X_test, y_test, num_samples=8):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        plt.figure(figsize=(15, 10))
        for i in range(num_samples):
            idx = np.random.randint(0, X_test.shape[0])
            
            plt.subplot(2, 4, i+1)
            probabilities = y_proba[idx]
            emotions = self.le.classes_
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
            bars = plt.bar(emotions, probabilities, color=colors)
            
            plt.title(f'True: {self.le.inverse_transform([y_test[idx]])[0]}\nPred: {self.le.inverse_transform([y_pred[idx]])[0]}')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def predict_emotion(self, model, audio_features=None):
        if audio_features is None:
            audio_features = np.random.normal(0, 1, 193)
        
        features_scaled = self.scaler.transform([audio_features])
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        emotion = self.le.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        return emotion, confidence, probabilities

def main():
    print("Emotion Recognition from Speech")
    print("=" * 50)
    
    emotion_recognition = EmotionRecognition()
    
    X, y = emotion_recognition.generate_synthetic_audio_features(n_samples=2000)
    
    results, X_test, y_test = emotion_recognition.train(X, y)
    
    best_model_name, best_model = emotion_recognition.evaluate(results, y_test)
    
    emotion_recognition.visualize_predictions(best_model, X_test, y_test)
    
    import joblib
    joblib.dump(best_model, 'emotion_recognition_model.pkl')
    joblib.dump(emotion_recognition.scaler, 'emotion_scaler.pkl')
    print(f"Model saved as 'emotion_recognition_model.pkl'")
    
    print("\nTesting with sample predictions...")
    for i in range(5):
        emotion, confidence, probabilities = emotion_recognition.predict_emotion(best_model)
        print(f"Prediction {i+1}: {emotion} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
