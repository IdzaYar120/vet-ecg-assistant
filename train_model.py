import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import pickle

def train():
    print("⏳ Завантаження датасету MIT-BIH...")
    try:
        df_train = pd.read_csv('mitbih_train.csv', header=None)
        df_test = pd.read_csv('mitbih_test.csv', header=None)
    except FileNotFoundError:
        print("❌ Помилка: Не знайдено файли csv. Скачайте їх з Kaggle (MIT-BIH Arrhythmia Database).")
        return

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    inp = layers.Input(shape=(187, 1))
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint('ecg_model.h5', save_best_only=True, monitor='val_accuracy')
    
    print("🚀 Починаємо навчання (це займе час)...")
    model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])
    
    print("✅ Модель навчено і збережено як 'ecg_model.h5'")

    classes = {0: 'Норма (N)', 1: 'Надшлуночкова (S)', 2: 'Шлуночкова (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

if __name__ == "__main__":
    train()