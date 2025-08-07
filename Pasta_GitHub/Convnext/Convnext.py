import os
import cv2
import numpy as np
import math
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler
)
from tensorflow.keras.applications.convnext import preprocess_input, ConvNeXtTiny
import tensorflow_addons as tfa

# Hyperparametros
DATA_DIR            = "/mnt/c/Pasta_GitHub/Dataset"
OUTPUT_DIR          = "output"
CSV_LOG_PATH        = os.path.join(OUTPUT_DIR, "training_log.csv")
OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "output.txt")

batch_size       = 8
num_classes      = 4
epochs           = 500
initial_lr       = 2e-4
clipnorm         = 0.5
weight_decay     = 1e-5
optimizer_choice = 1     # 1=AdamW,2=SGD,3=RAdam,4=NAdam
momentum         = 0.9
es_monitor       = 'val_mean_iou'
es_patience      = 40
warmup_epochs    = 3

# Dimensões
IMAGE_HEIGHT = 320
IMAGE_WIDTH  = 640

CLASS_NAMES = [
    'Preto absoluto - Sem Classificação',
    'Vermelho - Milho',
    'Amarelo - Panicum',
    'Verde - Solo'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# Caminhos
train_images = sorted(glob(os.path.join(DATA_DIR, "Training/Images/*")))
train_cats   = sorted(glob(os.path.join(DATA_DIR, "Training/Categories/*")))
train_masks  = sorted(glob(os.path.join(DATA_DIR, "Training/Category_ids/*")))
val_images   = sorted(glob(os.path.join(DATA_DIR, "Validation/Images/*")))
val_cats     = sorted(glob(os.path.join(DATA_DIR, "Validation/Categories/*")))
val_masks    = sorted(glob(os.path.join(DATA_DIR, "Validation/Category_ids/*")))
test_images  = sorted(glob(os.path.join(DATA_DIR, "Testing/Images/*")))
test_cats    = sorted(glob(os.path.join(DATA_DIR, "Testing/Categories/*")))
test_masks   = sorted(glob(os.path.join(DATA_DIR, "Testing/Category_ids/*")))

# Leitura e normalização
def read_rgb(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return tf.cast(img, tf.float32) / 127.5 - 1.0

def read_mask(path):
    m = tf.io.read_file(path)
    m = tf.image.decode_png(m, channels=1)
    return tf.cast(m, tf.int32)

def load_triplet(img_p, cat_p, msk_p):
    x   = read_rgb(img_p)
    cat = read_mask(cat_p)
    msk = read_mask(msk_p)
    return x, cat, msk

# Data augmentation
def augment_triplet(x, cat, msk):
    # flip horizontal/vertical
    if tf.random.uniform([]) < 0.5:
        x   = tf.image.flip_left_right(x)
        cat = tf.image.flip_left_right(cat)
        msk = tf.image.flip_left_right(msk)
    if tf.random.uniform([]) < 0.5:
        x   = tf.image.flip_up_down(x)
        cat = tf.image.flip_up_down(cat)
        msk = tf.image.flip_up_down(msk)
    # rotação em múltiplos de 90°
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x   = tf.image.rot90(x, k)
    cat = tf.image.rot90(cat, k)
    msk = tf.image.rot90(msk, k)
    # escala aleatória entre 0.8 e 1.2, então crop ou pad para 320×640
    scale = tf.random.uniform([], 0.8, 1.2)
    nh = tf.cast(tf.cast(IMAGE_HEIGHT, tf.float32) * scale, tf.int32)
    nw = tf.cast(tf.cast(IMAGE_WIDTH,  tf.float32) * scale, tf.int32)
    x   = tf.image.resize(x,   (nh, nw), method='bilinear')
    cat = tf.image.resize(cat, (nh, nw), method='nearest')
    msk = tf.image.resize(msk, (nh, nw), method='nearest')
    def _crop():
        xx   = tf.image.random_crop(x,   [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        cc   = tf.image.random_crop(cat, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        mm   = tf.image.random_crop(msk, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        return xx, cc, mm
    def _pad():
        xx   = tf.image.resize_with_crop_or_pad(x,   IMAGE_HEIGHT, IMAGE_WIDTH)
        cc   = tf.image.resize_with_crop_or_pad(cat, IMAGE_HEIGHT, IMAGE_WIDTH)
        mm   = tf.image.resize_with_crop_or_pad(msk, IMAGE_HEIGHT, IMAGE_WIDTH)
        return xx, cc, mm
    return tf.cond(tf.logical_and(nh >= IMAGE_HEIGHT, nw >= IMAGE_WIDTH), _crop, _pad)

# CutMix
def cutmix(batch_x, batch_y):
    B = tf.shape(batch_x)[0]
    idx = tf.random.shuffle(tf.range(B))

    H = IMAGE_HEIGHT
    W = IMAGE_WIDTH
    cut_ratio = 0.5

    cw = tf.cast(W * cut_ratio, tf.int32)
    ch = tf.cast(H * cut_ratio, tf.int32)
    cx = tf.random.uniform([], 0, W, dtype=tf.int32)
    cy = tf.random.uniform([], 0, H, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cw // 2, 0, W)
    y1 = tf.clip_by_value(cy - ch // 2, 0, H)
    x2 = tf.clip_by_value(cx + cw // 2, 0, W)
    y2 = tf.clip_by_value(cy + ch // 2, 0, H)

    # Máscara com canal = 1
    mask = tf.zeros([H, W, 1], tf.float32)
    mask = tf.tensor_scatter_nd_update(
        mask, [[y1, x1, 0]], [1.0]
    )
    mask = tf.pad(mask[y1:y2, x1:x2],
                  [[y1, H - y2], [x1, W - x2], [0, 0]])
    inv = 1.0 - mask

    px = tf.gather(batch_x, idx)
    py = tf.gather(batch_y, idx)

    mixed_x = batch_x * inv + px * mask
    mixed_y = tf.cast(batch_y, tf.float32) * inv + tf.cast(py, tf.float32) * mask
    mixed_y = tf.cast(tf.round(mixed_y), tf.int32)

    return mixed_x, mixed_y


# Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_cats,train_masks))
train_ds = train_ds.shuffle(len(train_images))
train_ds = train_ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augment_triplet, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda x,_,y: (x,y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.map(lambda x,y: cutmix(x,y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_images,val_cats,val_masks))
val_ds = val_ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x,_,y: (x,y), num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_cats,test_masks))
test_ds = test_ds.map(load_triplet, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x,_,y: (x,y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Loss combinado (SparseCategorical + Dice)
def dice_loss(y_true, y_pred, eps=1e-6):
    y_t = tf.squeeze(y_true, axis=-1)
    y_t = tf.one_hot(y_t, num_classes)
    y_p = tf.nn.softmax(y_pred, axis=-1)

    # Interseção e Soma
    inter = tf.reduce_sum(y_t * y_p, axis=[1,2])
    union = tf.reduce_sum(y_t + y_p, axis=[1,2])

    dice = (2. * inter + eps) / (union + eps)
    return 1. - tf.reduce_mean(dice, axis=-1)

def combined_loss(y_true, y_pred):
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    ce = tf.reduce_mean(ce, axis=[1, 2])
    dl = dice_loss(y_true, y_pred)
    return ce + dl

# SpacialDropout no ASPP
def convolution_block(x, filters=256, k=3, rate=1):
    x = layers.Conv2D(filters, k, dilation_rate=rate, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(x):
    s = tf.shape(x)
    y = tf.reduce_mean(x, axis=[1,2], keepdims=True)
    y = convolution_block(y, k=1)
    y = tf.image.resize(y, (s[1], s[2]), method='bilinear')

    b0 = convolution_block(x, k=1)
    b1 = convolution_block(x, k=3, rate=6)
    b2 = convolution_block(x, k=3, rate=12)
    b3 = convolution_block(x, k=3, rate=18)

    x = layers.Concatenate()([y,b0,b1,b2,b3])
    x = layers.SpatialDropout2D(0.1)(x)
    return convolution_block(x,k=1)

# Deeplabv3+ com ConvNext de backbone
def DeeplabV3Plus_ConvNeXt(num_classes):
    inp = keras.Input((None,None,3))
    x = (inp+1)*127.5
    x = preprocess_input(x)

    base = ConvNeXtTiny(weights='imagenet',include_preprocessing=False,include_top=False)

    x = base(x)
    x = DilatedSpatialPyramidPooling(x)
    x = tf.image.resize(x,tf.shape(inp)[1:3],'bilinear')
    logits = layers.Conv2D(num_classes,1,padding='same',kernel_initializer='he_normal')(x)
    return keras.Model(inputs=inp,outputs=logits)

model = DeeplabV3Plus_ConvNeXt(num_classes)
model.summary()

# Métricas e otimizadores
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

mean_iou = MeanIoUArgmaxNoBG(num_classes=num_classes, name='mean_iou')

if optimizer_choice==1:
    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay,
        learning_rate=initial_lr,
        clipnorm=clipnorm
    )
elif optimizer_choice==2:
    optimizer = keras.optimizers.SGD(
        learning_rate=initial_lr,
        momentum=momentum
    )
elif optimizer_choice==3:
    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=initial_lr
    )
else:
    optimizer = keras.optimizers.Nadam(
        learning_rate=initial_lr
    )

# Callback programado (Warm-Up de 3 épocas + decaimento cosseno)
def lr_fn(epoch):
    if epoch<warmup_epochs:
        return initial_lr*(epoch+1)/warmup_epochs
    e=epoch-warmup_epochs
    decay=epochs-warmup_epochs
    return initial_lr*0.5*(1+math.cos(math.pi*e/decay))

callbacks=[
    # CSVLogger para registro
    CSVLogger(CSV_LOG_PATH, separator=",", append=False),
    # Programado
    LearningRateScheduler(lr_fn, verbose=1),
    # EarlyStopping
    EarlyStopping(monitor=es_monitor, mode='max', patience=es_patience, restore_best_weights=True)
]

# Compilação do treinamento
model.compile(optimizer=optimizer,
              loss=combined_loss,
              metrics=[mean_iou, keras.metrics.SparseCategoricalAccuracy(name='accuracy')])

# Treino
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)

# Plotar e salvar os gráficos
def plot_hist(metric):
    tr, val = history.history[metric], history.history[f'val_{metric}']
    plt.figure()
    plt.plot(tr, label=f'train_{metric}')
    plt.plot(val, label=f'val_{metric}')
    plt.title(metric)
    plt.xlabel('época')
    plt.ylabel(metric)
    plt.legend()
    fn = os.path.join(OUTPUT_DIR, f"{metric}.png")
    plt.savefig(fn,bbox_inches='tight')
    plt.close()

for m in ['loss','accuracy','mean_iou']:
    plot_hist(m)

# Perda e acurácia no Teste
test_loss, test_miou, test_acc = model.evaluate(test_ds, verbose=1)

# --- Avaliação do Test Set sem TTA (normal IoU, classes 1..3) ---
cm = np.zeros((num_classes, num_classes), dtype=int)
for imgs, msks in test_ds:
    # predição normal
    logits = model.predict(imgs)
    preds  = np.argmax(logits, axis=-1).flatten()
    gts    = msks.numpy().flatten()
    # acumula confusion matrix
    cm += tf.math.confusion_matrix(
        labels=gts,
        predictions=preds,
        num_classes=num_classes
    ).numpy()

# calcula IoU apenas para classes 1,2,3
ious_no = [
    cm[i, i] / (
      cm[i, i]
      + cm[:, i].sum() - cm[i, i]
      + cm[i, :].sum() - cm[i, i]
      + 1e-8
    )
    for i in range(1, num_classes)
]

print("\n--- IoU por classe (teste, sem TTA) ---")
for idx, v in enumerate(ious_no, start=1):
    print(f"Classe {idx} ({CLASS_NAMES[idx]}): {v:.4f}")
print("→ mIoU global (teste):", np.mean(ious_no))

# Escrita das métricas no meu arquivo de saída
train_loss, train_acc, train_miou = (
    history.history['loss'][-1],
    history.history['accuracy'][-1],
    history.history['mean_iou'][-1]
)
val_loss, val_acc, val_miou = (
    history.history['val_loss'][-1],
    history.history['val_accuracy'][-1],
    history.history['val_mean_iou'][-1]
)


# --- Gravação no arquivo de saída ---
with open(OUTPUT_METRICS_FILE, "w") as f:
    f.write("=== Métricas de Treino e Validação ===\n")
    f.write(
        f"Train → loss: {train_loss:.4f}    "
        f"acc: {train_acc:.4f}    mIoU: {train_miou:.4f}\n"
    )
    f.write(
        f"Val   → loss: {val_loss:.4f}    "
        f"acc: {val_acc:.4f}    mIoU: {val_miou:.4f}\n\n"
    )
    f.write("=== Métricas no Test Set (loss/acc) ===\n")
    f.write(
        f"Test  → loss: {test_loss:.4f}    "
        f"acc: {test_acc:.4f}    "
        f"mIoU: {test_miou:.4f}\n\n"
    )
    f.write("=== IoU por Classe (Teste, sem TTA) ===\n")
    for idx, iou in enumerate(ious_no, start=1):
        f.write(f"Classe {idx} ({CLASS_NAMES[idx]}): {iou:.4f}\n")
    f.write(f"\n→ mIoU global (teste): {np.mean(ious_no):.4f}\n")

# --- IoU por imagem (Test Set) ---
with open(OUTPUT_METRICS_FILE, "a") as f:
    f.write("\n--- IoU por imagem (test) ---\n")

for img_path, msk_path in zip(test_images, test_masks):
    raw     = read_rgb(img_path).numpy()
    msk_gt  = read_mask(msk_path).numpy().squeeze(-1)
    pred    = model.predict(raw[None, ...])[0]
    msk_pred= np.argmax(pred, axis=-1)

    # IoU por classe
    unique = np.unique(msk_gt)
    ious_img = []
    for c in range(num_classes):
        tp = np.sum((msk_pred == c) & (msk_gt == c))
        fp = np.sum((msk_pred == c) & (msk_gt != c))
        fn = np.sum((msk_pred != c) & (msk_gt == c))
        ious_img.append(tp / (tp + fp + fn + 1e-8))

    # mIoU por imagem, apenas classes 1..3
    present = [c for c in unique if c != 0]
    miou_img = np.mean([ious_img[c] for c in present]) if present else 0.0

    line = (
        f"{os.path.basename(img_path)} → IoUs: "
        f"{[f'{x:.4f}' for x in ious_img]}, mIoU: {miou_img:.4f}\n"
    )
    print(line.strip())
    with open(OUTPUT_METRICS_FILE, "a") as f:
        f.write(line)

    # salva máscara, colormap e overlay em 'test'
    mat   = loadmat(os.path.join(DATA_DIR, 'human_colormap.mat'))
    cmap  = (mat['colormap'] * 100).astype(np.uint8)
    clr   = np.zeros((*msk_pred.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        clr[msk_pred == c] = cmap[c]
    ovl   = cv2.addWeighted(
        ((raw + 1) * 127.5).astype(np.uint8),
        0.35, clr, 0.65, 0
    )
    base  = os.path.splitext(os.path.basename(img_path))[0]
    odir  = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(odir, exist_ok=True)
    cv2.imwrite(f"{odir}/{base}_mask.png",   msk_pred.astype(np.uint8))
    cv2.imwrite(f"{odir}/{base}_color.png",  cv2.cvtColor(clr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{odir}/{base}_overlay.png",cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))

print("Acabou sem erros, resultados em", OUTPUT_DIR)