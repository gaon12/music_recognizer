import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import hashlib
import json

def check_dataset_changes(csv_path, dataset_hash_file):
    with open(csv_path, 'rb') as f:
        current_hash = hashlib.md5(f.read()).hexdigest()

    if os.path.exists(dataset_hash_file):
        with open(dataset_hash_file, 'r') as f:
            previous_hash = f.read()
    else:
        previous_hash = ""

    if current_hash != previous_hash:
        with open(dataset_hash_file, 'w') as f:
            f.write(current_hash)
        return True
    else:
        return False


nltk.download('stopwords')

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return np.empty((40,))
    return mfccs_processed

def get_class_weights(y):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y, axis=1)), np.argmax(y, axis=1))
    return class_weights

def train_model(existing_model, x, y, epochs, batch_size):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    class_weights = get_class_weights(y)
    
    if existing_model is None:
        model = create_1d_cnn_model(input_shape=x.shape[1:])
    else:
        model = existing_model
    
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)
    
    return model, history

def train_word2vec_model(text_data, model_path, epochs=100):
    model = Word2Vec(sentences=text_data, vector_size=100, window=5, min_count=1, workers=4)
    model.train(text_data, total_examples=model.corpus_count, epochs=epochs)
    model.wv.save_word2vec_format(model_path, binary=True)
    return model.wv

def preprocess_text_data(text_data, max_len, embedding_model):
    processed_data = []
    for row in text_data:
        processed_row = []
        tokenized_row = [word for word in row if isinstance(word, str) and word.lower() not in stop_words]  # 빈 리스트([]) 제거
        tokenized_row = np.array([embedding_model[word] for word in tokenized_row if word in embedding_model])
        processed_row.append(tokenized_row)
        processed_data.append(np.concatenate(processed_row))

    processed_data = pad_sequences(processed_data, maxlen=max_len, dtype='float32', padding='post')
    
    return processed_data

def load_data(csv_path, embedding_model):
    max_len = 100
    data = pd.read_csv(csv_path)
    file_names = data["파일명"].values
    genres = data["장르"].values
    unique_genres = np.unique(genres)

    text_data = data[["제목", "작곡가", "작사가", "    앨범명", "발매연도", "아티스트", "커버 아티스트명", "기타 정보"]].values
    text_data = [list(row) for row in text_data]  # 2차원 리스트로 변환
    text_data = [[" ".join(row)] for row in text_data]

    # Tokenize and remove stopwords
    text_data = [[word.lower() for word in tokenizer.tokenize(" ".join(row)) if word.lower() not in stop_words] for row in text_data]

    text_data = preprocess_text_data(text_data, max_len, embedding_model)

    labels = {genre: i for i, genre in enumerate(unique_genres)}
    num_classes = len(unique_genres)

    features = []
    labels_encoded = []
    for file_name, genre in zip(file_names, genres):
        feature = extract_features(file_name)
        if np.any(np.isnan(feature)):
            print(f"Skipping file: {file_name} due to error")
            continue
        features.append(feature)
        labels_encoded.append(labels[genre])

    x = np.array(features)
    y = tf.keras.utils.to_categorical(labels_encoded, num_classes=num_classes)

    return x, y, labels, text_data

def create_1d_cnn_model(num_classes, embedding_model, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((40, 1), input_shape=(40,)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_or_train_word2vec_model(csv_path, word2vec_model_path):
    if not os.path.exists(word2vec_model_path):
        os.makedirs(os.path.dirname(word2vec_model_path), exist_ok=True)
        data = pd.read_csv(csv_path)
        text_data = data[["제목", "작곡가", "작사가", "앨범명", "발매연도", "아티스트", "커버 아티스트명", "기타 정보"]].values
        text_data = [list(row) for row in text_data]  # 2차원 리스트로 변환
        text_data = [[" ".join(row)] for row in text_data]
        text_data = [[word.lower() for word in tokenizer.tokenize(" ".join(row)) if word.lower() not in stop_words] for row in text_data]
        embedding_model = train_word2vec_model(text_data, word2vec_model_path)
    else:
        embedding_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, unicode_errors='ignore')
    
    return embedding_model

def train_and_save_models(csv_path, epochs, batch_size, incremental_learning=False):
    study_files_path = "study_files"
    if not os.path.exists(study_files_path):
        os.makedirs(study_files_path)

    word2vec_model_path = os.path.join(study_files_path, 'mai_model.bin')
    embedding_model = load_or_train_word2vec_model(csv_path, word2vec_model_path)

    x, y, labels, text_data = load_data(csv_path, embedding_model)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    max_len = 100
    text_data_train, text_data_test = train_test_split(text_data, test_size=0.2, random_state=42)
    x_train_text = preprocess_text_data(text_data_train, max_len, embedding_model)
    x_test_text = preprocess_text_data(text_data_test, max_len, embedding_model)

    model_count = 0
    while True:
        model_file_path = os.path.join(study_files_path, f'music_genre_model_{model_count + 1}.h5')
        if not os.path.exists(model_file_path):
            break
        model_count += 1

    num_classes = len(labels)

    # Load the last saved model if incremental_learning is True, otherwise create a new model
    if incremental_learning and model_count > 0:
        model_file_path = os.path.join(study_files_path, f'music_genre_model_{model_count}.h5')
        model = tf.keras.models.load_model(model_file_path)
    else:
        model = create_1d_cnn_model(num_classes, embedding_model, max_length=20)

    history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    # Save the updated model
    model.save(model_file_path)

    return model_file_path

def create_1d_cnn_model(num_classes, embedding_model, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((40, 1), input_shape=(40,)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_or_train_word2vec_model(csv_path, word2vec_model_path):
    if not os.path.exists(word2vec_model_path):
        os.makedirs(os.path.dirname(word2vec_model_path), exist_ok=True)
        data = pd.read_csv(csv_path)
        text_data = data[["제목", "작곡가", "작사가", "앨범명", "발매연도", "아티스트", "커버 아티스트명", "기타 정보"]].values
        text_data = [list(row) for row in text_data]  # 2차원 리스트로 변환
        text_data = [[" ".join(row)] for row in text_data]
        text_data = [[word.lower() for word in tokenizer.tokenize(" ".join(row)) if word.lower() not in stop_words] for row in text_data]
        embedding_model = train_word2vec_model(text_data, word2vec_model_path)
    else:
        embedding_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True, unicode_errors='ignore')
    
    return embedding_model

def train_and_save_models(csv_path, epochs, batch_size):
    study_files_path = "study_files"
    if not os.path.exists(study_files_path):
        os.makedirs(study_files_path)

    word2vec_model_path = os.path.join(study_files_path, 'mai_model.bin')
    embedding_model = load_or_train_word2vec_model(csv_path, word2vec_model_path)

    x, y, labels, text_data = load_data(csv_path, embedding_model)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    max_len = 100
    text_data_train, text_data_test = train_test_split(text_data, test_size=0.2, random_state=42)
    x_train_text = preprocess_text_data(text_data_train, max_len, embedding_model)
    x_test_text = preprocess_text_data(text_data_test, max_len, embedding_model)

    model_count = 0
    while True:
        model_file_path = os.path.join(study_files_path, f'music_genre_model_{model_count + 1}.h5')
        if not os.path.exists(model_file_path):
            break
        model_count += 1

        num_classes = len(labels)
    model = create_1d_cnn_model(num_classes, embedding_model, max_length=20)
    history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

    model.save(model_file_path)

    return model_file_path

def get_all_model_paths(study_files_path):
    model_paths = []
    for file in os.listdir(study_files_path):
        if file.endswith('.h5'):
            model_paths.append(os.path.join(study_files_path, file))
    return model_paths

csv_path = "std.csv"
study_files_path = "study_files"
word2vec_model_path = os.path.join(study_files_path, 'mai_model.bin')
embedding_model = load_or_train_word2vec_model(csv_path, word2vec_model_path)  # Add this line to load the embedding_model
x, y, labels, text_data = load_data(csv_path, embedding_model)  # Pass the embedding_model to load_data
epochs = 50
batch_size = 32

# Train and save the models
model_file_path = train_and_save_models(csv_path, epochs, batch_size)

# the models with incremental learning option
dataset_hash_file = os.path.join(study_files_path, 'dataset_hash.txt')
incremental_learning = not check_dataset_changes(csv_path, dataset_hash_file)
model_file_path = train_and_save_models(csv_path, epochs, batch_size, incremental_learning=incremental_learning)

# Data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Get all the model paths
study_files_path = "study_files"
model_paths = get_all_model_paths(study_files_path)

def ensemble_prediction(model_paths, x):
    predictions = []

    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        prediction = model.predict(x)
        predictions.append(prediction)

    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# Use the ensemble of models for prediction
y_pred = ensemble_prediction(model_paths, x_test)

# Calculate accuracy score
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print("Ensemble model accuracy: {:.2f}%".format(accuracy * 100))
