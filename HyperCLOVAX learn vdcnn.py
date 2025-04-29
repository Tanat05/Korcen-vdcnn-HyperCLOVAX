import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Layer, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "물리적 GPU,", len(logical_gpus), "논리적 GPU")
    except RuntimeError as e:
        print(e)
else:
    print("GPU 없음")

try:
    data = np.load('data/encoded_data HyperCLOVAX/all_encoded_data.npz')

    encoded_texts = data["encoded_texts"]
    maxlen = encoded_texts.shape[1]
    encoded_labels = data["encoded_labels"]

    print(f"maxlen: {maxlen}")

except FileNotFoundError:
    print("오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

num_pos = np.sum(encoded_labels == 1)
num_neg = len(encoded_labels) - num_pos

print(f"label 1 개수: {num_pos}")
print(f"label 0 개수: {num_neg}")
if num_pos > 0:
    print(f"비율: 1 : {num_neg/num_pos:.2f}")
else:
    print("비율: label 1 개수가 0입니다.")

weight_0 = 1
weight_1 = 6
class_weight = {0: weight_0, 1: weight_1}
print(f"클래스 가중치: {class_weight}")

X_train_val, X_test, y_train_val, y_test = train_test_split(encoded_texts, encoded_labels, test_size=0.2, random_state=62, stratify=encoded_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2/0.8, random_state=44, stratify=y_train_val)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("학습, 검증, 테스트 데이터 분할 완료")

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = Attention()
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        return self.attention([x, x, x])

    def compute_output_shape(self, input_shape):
        return input_shape

inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=110524, output_dim=8)(inputs)

conv_list = []
filter_sizes = [2, 4, 8]
for filter_size in filter_sizes:
    x_conv = x
    x_conv = Conv1D(filters=64, kernel_size=filter_size, activation='relu')(x_conv)
    x_conv = MaxPooling1D(pool_size=2)(x_conv)
    x_conv = SelfAttention()(x_conv)
    pool = GlobalMaxPooling1D()(x_conv)
    conv_list.append(pool)

x = tf.keras.layers.concatenate(conv_list, axis=-1)

x = Dense(32, activation='selu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.4)(x)
x = Dense(16, activation='selu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.25)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

filepath = "vdcnn_model_best HyperCLOVAX.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=1231)

history_list = []
fold_results = []

print("\nK-Fold 교차 검증 시작...")
for i, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'\n폴드 {i + 1}/{k}')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    history = model.fit(X_train[train_idx], y_train[train_idx],
                        validation_data=(X_train[val_idx], y_train[val_idx]),
                        epochs=100,
                        batch_size=2048,
                        callbacks=[checkpoint, early_stopping],
                        class_weight=class_weight,
                        verbose=1)

    history_list.append(history.history)

    print(f"폴드 {i+1} 검증 데이터 평가 중...")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_train[val_idx], y_train[val_idx], verbose=0)
    fold_results.append({
        'fold': i + 1,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall
    })
    print(f"폴드 {i+1} - 검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}, 검증 정밀도: {val_precision:.4f}, 검증 재현율: {val_recall:.4f}")

print("\nK-Fold 교차 검증 완료.")
print("폴드 결과:")
for result in fold_results:
    print(f"폴드 {result['fold']}: 검증 손실={result['val_loss']:.4f}, 검증 정확도={result['val_accuracy']:.4f}, 검증 정밀도={result['val_precision']:.4f}, 검증 재현율={result['val_recall']:.4f}")

avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
avg_val_acc = np.mean([r['val_accuracy'] for r in fold_results])
avg_val_precision = np.mean([r['val_precision'] for r in fold_results])
avg_val_recall = np.mean([r['val_recall'] for r in fold_results])

print(f"\n{k}개 폴드 전체의 평균 검증 메트릭:")
print(f"평균 검증 손실: {avg_val_loss:.4f}")
print(f"평균 검증 정확도: {avg_val_acc:.4f}")
print(f"평균 검증 정밀도: {avg_val_precision:.4f}")
print(f"평균 검증 재현율: {avg_val_recall:.4f}")

print("\n학습 히스토리 플롯 (폴드 전체 연결)...")
try:
    history = {}
    max_epochs = max([len(h['loss']) for h in history_list])

    for key in history_list[0].keys():
        padded_histories = []
        for h in history_list:
            padding_needed = max_epochs - len(h[key])
            padded_data = np.pad(h[key], (0, padding_needed), 'constant', constant_values=np.nan)
            padded_histories.append(padded_data)
        history[key] = np.nanmean(padded_histories, axis=0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='학습 (평균)')
    plt.plot(history['val_accuracy'], label='검증 (평균)')
    plt.title('모델 정확도 (폴드 전체 평균)')
    plt.ylabel('정확도')
    plt.xlabel('에포크')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='학습 (평균)')
    plt.plot(history['val_loss'], label='검증 (평균)')
    plt.title('모델 손실 (폴드 전체 평균)')
    plt.ylabel('손실')
    plt.xlabel('에포크')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vdcnn_training_history_avg.png')
    plt.show()

except Exception as e:
    print(f"히스토리 플롯 오류: {e}")
    print("히스토리 플롯 건너뛰기.")

print("\n초기 검증 데이터 (X_val, y_val) 평가 중...")
val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
val_pred_proba = model.predict(X_val)
val_pred_labels = (val_pred_proba > 0.5).astype(int)
val_f1 = f1_score(y_val, val_pred_labels)
val_auc = roc_auc_score(y_val, val_pred_proba)

print(f'검증 손실: {val_loss:.4f}')
print(f'검증 정확도: {val_acc:.4f}')
print(f'검증 정밀도: {val_precision:.4f}')
print(f'검증 재현율: {val_recall:.4f}')
print(f'검증 F1-Score: {val_f1:.4f}')
print(f'검증 AUC: {val_auc:.4f}')

print("\n독립적인 테스트 데이터 (X_test, y_test) 평가 중...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
test_pred_proba = model.predict(X_test)
test_pred_labels = (test_pred_proba > 0.5).astype(int)
test_f1 = f1_score(y_test, test_pred_labels)
test_auc = roc_auc_score(y_test, test_pred_proba)

print(f'테스트 손실: {test_loss:.4f}')
print(f'테스트 정확도: {test_acc:.4f}')
print(f'테스트 정밀도: {test_precision:.4f}')
print(f'테스트 재현율: {test_recall:.4f}')
print(f'테스트 F1-Score: {test_f1:.4f}')
print(f'테스트 AUC: {test_auc:.4f}')
