import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt

# Tạo thư mục lưu model nếu chưa có
os.makedirs('utils', exist_ok=True)

# Load dữ liệu
print("🔄 Đang load dữ liệu...")
data = np.load('sign_language_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
joblib.dump(label_encoder, 'utils/label_encoder.pkl')

# One-hot encoding
num_classes = len(np.unique(y_train))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"📊 Kích thước dữ liệu:")
print(f"Training: {X_train.shape}")
print(f"Testing: {X_test.shape}")

# Xây dựng mô hình đơn giản và hiệu quả
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Tóm tắt mô hình
model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_accuracy'
    ),
    ModelCheckpoint(
        'utils/best_model.keras',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

print("\n🚀 Bắt đầu training...")

# Training
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

# Đánh giá trên tập test
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"\n✅ Accuracy trên tập kiểm tra: {test_acc:.4f}")

# Lưu mô hình
model.save('utils/sign_language_model.keras')
print("✅ Đã lưu mô hình tại utils/sign_language_model.keras")

# Vẽ biểu đồ huấn luyện
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Biểu đồ Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Biểu đồ Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("utils/training_history.png")
print("📊 Đã lưu biểu đồ tại utils/training_history.png")
