import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from joblib import load
from tensorflow.keras import layers,Model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mob_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess

DATA_DIR = "Skin Disease Dataset"  

classes = sorted(os.listdir(DATA_DIR))
num_classes = len(classes)
# -------------------------------
# Channel Attention
# -------------------------------
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_dense_1 = layers.Dense(channel // ratio, activation='relu')
    shared_dense_2 = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    attention = layers.Activation('sigmoid')(avg_pool + max_pool)
    return layers.Multiply()([input_feature, attention])


# -------------------------------
# Spatial Attention (FIXED)
# -------------------------------
def spatial_attention(input_feature):
    avg_pool = layers.Lambda(
        lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
    )(input_feature)

    max_pool = layers.Lambda(
        lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
    )(input_feature)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding='same',
        activation='sigmoid'
    )(concat)

    return layers.Multiply()([input_feature, attention])


# -------------------------------
# CBAM Block
# -------------------------------
def cbam_block(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x


from tensorflow.keras.applications import EfficientNetB0
def build_efficientnet_cbam(num_classes):

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base_model.input, output)

from tensorflow.keras.applications import MobileNetV3Large
def build_mobilenetv3_cbam(num_classes):

    base_model = MobileNetV3Large(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base_model.input, output)

from tensorflow.keras.applications import ResNet50
def build_resnet_cbam(num_classes):

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_classes, activation="softmax")(x)
    return Model(base_model.input, output)

efficientnet_model = build_efficientnet_cbam(num_classes)
efficientnet_model.load_weights("Weights/CBAM_EfficientNetB0_model_final.h5")

mobilenet_model = build_mobilenetv3_cbam(num_classes)
mobilenet_model.load_weights("Weights/CBAM_mobilenetv3_model_final.h5")

resnet_model = build_resnet_cbam(num_classes)
resnet_model.load_weights("Weights/CBAM_ResNet50_model_final.h5")

meta_model = load("Weights/hybrid_cbam_meta_model.pkl")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

def convert_to_logits_model(model):
    x = model.layers[-2].output
    logits = Dense(model.layers[-1].units, activation=None)(x)
    logits_model = Model(model.input, logits)
    logits_model.layers[-1].set_weights(model.layers[-1].get_weights())
    return logits_model

eff_logits_model = convert_to_logits_model(efficientnet_model)
mob_logits_model = convert_to_logits_model(mobilenet_model)
res_logits_model = convert_to_logits_model(resnet_model)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    return {
        "eff": eff_preprocess(img.copy()),
        "mob": mob_preprocess(img.copy()),
        "res": res_preprocess(img.copy())
    }
def predict_skin_disease(img_path):

    imgs = preprocess_image(img_path)

    # logits
    l1 = eff_logits_model.predict(imgs["eff"], verbose=0)
    l2 = mob_logits_model.predict(imgs["mob"], verbose=0)
    l3 = res_logits_model.predict(imgs["res"], verbose=0)

    # probabilities
    p1 = tf.nn.softmax(l1).numpy()
    p2 = tf.nn.softmax(l2).numpy()
    p3 = tf.nn.softmax(l3).numpy()

    # dual stacking features (36)
    stacked_features = np.hstack([l1, l2, l3, p1, p2, p3])

    final_pred = meta_model.predict(stacked_features)

    predicted_class = int(final_pred[0])
    confidence = float(np.max(p2) * 100)

    return predicted_class, round(confidence, 2)


