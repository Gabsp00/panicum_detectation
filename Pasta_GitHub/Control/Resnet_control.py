import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, TerminateOnNaN, ReduceLROnPlateau, EarlyStopping

# Hiperparametros e Dimensões
IMAGE_SIZE  = 512
BATCH_SIZE  = 8
NUM_CLASSES = 4

# Diretórios
DATA_DIR    = os.path.normpath(
    r"/mnt/c/Gabslau/TCC/IC_Andre/Teste_Keras_MDTS/instance-level_human_parsing/instance-level_human_parsing"
)
os.makedirs("prediction_outputs", exist_ok=True)
OUTPUT_CSV_LOG     = os.path.join("prediction_outputs", "training_log.csv")
OUTPUT_METRICS_TXT = os.path.join("prediction_outputs", "output.txt")

# GPU
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Leitura e normalização
def read_image(path, mask=False):
    img = tf.io.read_file(path)
    if mask:
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="nearest")
    else:
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE], method="bilinear")
        img = (img / 127.5) - 1.0
    return img

def load_data(img_path, mask_path):
    img = read_image(img_path, mask=False)
    msk = read_image(mask_path, mask=True)
    return img, msk

def make_dataset(image_list, mask_list):
    ds = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    ds = ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Caminhos
train_images = sorted(glob(os.path.join(DATA_DIR, "Training/Images/*")))
train_masks  = sorted(glob(os.path.join(DATA_DIR, "Training/Category_ids/*")))
val_images   = sorted(glob(os.path.join(DATA_DIR, "Validation/Images/*")))
val_masks    = sorted(glob(os.path.join(DATA_DIR, "Validation/Category_ids/*")))
test_images  = sorted(glob(os.path.join(DATA_DIR, "Testing/Images/*")))
test_masks   = sorted(glob(os.path.join(DATA_DIR, "Testing/Category_ids/*")))

# Datasets
train_ds = make_dataset(train_images, train_masks)
val_ds   = make_dataset(val_images, val_masks)
test_ds  = make_dataset(test_images, test_masks)

# Métricas
class MeanIoUArgmaxNoBG(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        # Matriz de confusão acumulada
        self.cm = self.add_weight(
            'confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.int32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        cm_batch = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=self.cm.dtype
        )
        self.cm.assign_add(cm_batch)

    def result(self):
        # Cálculo do IoU para as classes 1, 2 e 3
        cm = tf.cast(self.cm, tf.float32)
        ious = []
        for i in range(1, self.num_classes):
            tp = cm[i, i]
            fp = tf.reduce_sum(cm[:, i]) - tp
            fn = tf.reduce_sum(cm[i, :]) - tp
            ious.append(tp / (tp + fp + fn + 1e-8))
        return tf.reduce_mean(tf.stack(ious))

    def reset_states(self):
        tf.keras.backend.set_value(
            self.cm,
            np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        )

# Deeplabv3+ com Resnet de backbone (original, sem melhorias)
def convolution_block(x, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
    x = layers.Conv2D(
        num_filters,
        kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

# ASPP
def DilatedSpatialPyramidPooling(x):
    dims = x.shape
    pool = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(x)
    pool = convolution_block(pool, kernel_size=1, use_bias=True)
    pool = layers.UpSampling2D(
        size=(dims[-3] // pool.shape[1], dims[-2] // pool.shape[2]),
        interpolation="bilinear",
    )(pool)
    c1  = convolution_block(x, kernel_size=1, dilation_rate=1)
    c6  = convolution_block(x, kernel_size=3, dilation_rate=6)
    c12 = convolution_block(x, kernel_size=3, dilation_rate=12)
    c18 = convolution_block(x, kernel_size=3, dilation_rate=18)
    x   = layers.Concatenate()([pool, c1, c6, c12, c18])
    return convolution_block(x, kernel_size=1)

def DeeplabV3Plus(image_size, num_classes):
    inp  = keras.Input(shape=(image_size, image_size, 3))
    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=inp
    )
    x    = base.get_layer("conv4_block6_2_relu").output
    x    = DilatedSpatialPyramidPooling(x)
    x    = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    skip = base.get_layer("conv2_block3_2_relu").output
    skip = convolution_block(skip, num_filters=48, kernel_size=1)
    x    = layers.Concatenate()([x, skip])
    x    = convolution_block(x)
    x    = convolution_block(x)
    x    = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    out  = layers.Conv2D(num_classes, kernel_size=1, padding="same")(x)
    return keras.Model(inputs=inp, outputs=out)

model = DeeplabV3Plus(IMAGE_SIZE, NUM_CLASSES)

# Compilações
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
loss_fn   = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
mean_iou = MeanIoUArgmaxNoBG(num_classes=NUM_CLASSES, name='mean_iou')

# Compilação do treinamento
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=[mean_iou, keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
)
model.summary()

# Callbacks
# CSVLogger para registro
csv_logger = CSVLogger(OUTPUT_CSV_LOG, separator=",", append=False)
# EarlyStopping
early_stop = EarlyStopping(
    monitor="val_mean_iou",
    mode="max",
    patience=50,
    verbose=1,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

callbacks = [TerminateOnNaN(), reduce_lr, early_stop, csv_logger]

# Treino
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=callbacks,
)

# Plotar e salvar os gráficos
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.title("Loss"); plt.savefig("prediction_outputs/loss.png"); plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.title("Accuracy"); plt.savefig("prediction_outputs/accuracy.png"); plt.close()

plt.figure()
plt.plot(history.history["mean_iou"], label="train_miou")
plt.plot(history.history["val_mean_iou"], label="val_miou")
plt.legend(); plt.title("Mean IoU"); plt.savefig("prediction_outputs/mean_iou.png"); plt.close()

# Métricas do treino e validação
train_loss = history.history["loss"][-1]
train_acc  = history.history["accuracy"][-1]
train_miou = history.history["mean_iou"][-1]

val_loss = history.history["val_loss"][-1]
val_acc  = history.history["val_accuracy"][-1]
val_miou = history.history["val_mean_iou"][-1]

# Avaliação do Teste (sem TTA)
res_test = model.evaluate(test_ds, verbose=1)
test_loss = res_test[0]
test_acc  = res_test[1]
test_miou = res_test[2]

# Cálculo de IoU por classe no Teste
cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for imgs, msks in test_ds:
    p     = model.predict(imgs)
    preds = np.argmax(p, axis=-1).flatten()
    gts   = msks.numpy().flatten()
    cm_test += tf.math.confusion_matrix(
        labels=gts, predictions=preds, num_classes=NUM_CLASSES
    ).numpy()

# IoU por classe
ious_no = []
for i in range(1, NUM_CLASSES):
    tp = cm_test[i, i]
    fp = cm_test[:, i].sum() - tp
    fn = cm_test[i, :].sum() - tp
    ious_no.append(tp / (tp + fp + fn + 1e-8))

# Cálculo
per_image_results = []
for img_path, msk_path in zip(test_images, test_masks):
    raw    = read_image(img_path, mask=False).numpy()
    msk_gt = read_image(msk_path, mask=True).numpy().squeeze(-1)

    logits   = model.predict(raw[np.newaxis, ...])[0]
    msk_pred = np.argmax(logits, axis=-1)

    unique = np.unique(msk_gt)
    ious_img = []
    for c in range(NUM_CLASSES):
        tp = np.sum((msk_pred == c) & (msk_gt == c))
        fp = np.sum((msk_pred == c) & (msk_gt != c))
        fn = np.sum((msk_pred != c) & (msk_gt == c))
        ious_img.append(tp / (tp + fp + fn + 1e-8))

    present = [c for c in unique if c != 0]
    miou_img = np.mean([ious_img[c] for c in present]) if present else 0.0
    per_image_results.append((os.path.basename(img_path), ious_img, miou_img))

# Escrita das métricas no meu arquivo de saída
with open(OUTPUT_METRICS_TXT, "w") as f:
    f.write("=== Train & Val Metrics ===\n")
    f.write(f"Train → loss: {train_loss:.4f}  acc: {train_acc:.4f}  mIoU: {train_miou:.4f}\n")
    f.write(f"Val   → loss: {val_loss:.4f}  acc: {val_acc:.4f}  mIoU: {val_miou:.4f}\n\n")

    f.write("=== Test Set Metrics (sem classe 0) ===\n")
    f.write(f"Test  → loss: {test_loss:.4f}  acc: {test_acc:.4f}  mIoU: {test_miou:.4f}\n\n")

    f.write("--- IoU por Classe (1..3) ---\n")
    for idx, iou in enumerate(ious_no, start=1):
        f.write(f"Classe {idx}: {iou:.4f}\n")
    f.write("\n")

    f.write("--- IoU por Imagem (Test Set) ---\n")
    for name, ious_img, miou_img in per_image_results:
        f.write(
            f"{name} → IoUs: "
            + "[" + ", ".join(f"{x:.4f}" for x in ious_img) + "]"
            + f", mIoU (presentes): {miou_img:.4f}\n"
        )
print(f"Arquivo de métricas salvo em {OUTPUT_METRICS_TXT}")

# Salva a máscara, o colormap e o overlay
mat      = loadmat(os.path.join(DATA_DIR, "human_colormap.mat"))
colormap = (mat["colormap"] * 100).astype(np.uint8)

def infer(img_tensor):
    logits = model.predict(tf.expand_dims(img_tensor, 0))
    return np.argmax(logits[0], axis=-1)

def decode_segmentation(mask):
    h, w = mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = colormap[c]
    return rgb

def save_predictions(image_paths, folder):
    os.makedirs(folder, exist_ok=True)
    for p in image_paths:
        raw = read_image(p, mask=False).numpy()
        m   = infer(raw)
        col = decode_segmentation(m)
        ovl = cv2.addWeighted(
            (raw * 127.5 + 127.5).astype(np.uint8), 0.35, col, 0.65, 0
        )
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(f"{folder}/{base}_mask.png", m.astype(np.uint8))
        cv2.imwrite(f"{folder}/{base}_color.png", cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            f"{folder}/{base}_overlay.png", cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR)
        )

# Agora salva para TODAS as imagens de teste
save_predictions(test_images,  "outputs/test")
