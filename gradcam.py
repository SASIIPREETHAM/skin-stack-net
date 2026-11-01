"""import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


# ---------------------------------------------------------
# Generic GradCAM function
# ---------------------------------------------------------
def compute_gradcam(img_array, model, preprocess_fn, last_conv_layer=None):

    img_array = preprocess_fn(img_array)

    # auto detect last conv layer if not given
    if last_conv_layer is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

    conv_layer = model.get_layer(last_conv_layer)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val != 0:
        heatmap /= max_val

    return heatmap.numpy()


# ---------------------------------------------------------
# WEIGHTED HYBRID GRADCAM
# ---------------------------------------------------------
def generate_weighted_hybrid_gradcam(
        img_path,
        eff_model,
        mob_model,
        res_model,
        output_path
):
   

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized.astype("float32"), axis=0)

    # preprocess functions
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mob_pre
    from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre

    # compute individual heatmaps
    heat_eff = compute_gradcam(img_array.copy(), eff_model, eff_pre)
    heat_mob = compute_gradcam(img_array.copy(), mob_model, mob_pre)
    heat_res = compute_gradcam(img_array.copy(), res_model, res_pre)

    # -------------------------------------------------
    # WEIGHTED FUSION (IMPORTANT)
    # -------------------------------------------------
    w_eff = 0.35
    w_mob = 0.20
    w_res = 0.45

    heatmap = (w_eff * heat_eff) + (w_mob * heat_mob) + (w_res * heat_res)

    # normalize again
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # resize to original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # overlay
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # save
    cv2.imwrite(output_path, superimposed_img)

    return output_path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


def generate_resnet50_gradcam(img_path, model, output_path):

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized.astype("float32"), axis=0)

    img_array = preprocess_input(img_array)

    # IMPORTANT: fixed last conv layer for ResNet50
    last_conv_layer_name = "conv5_block3_out"

    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()

    # Resize + overlay
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, superimposed_img)

    return output_path
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input


def generate_resnet50_gradcam(img_path, model, output_path):
    """
    FINAL GradCAM++ for ResNet50 + CBAM architecture
    """

    # -------------------------------------------------
    # 1. Load image
    # -------------------------------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Invalid image path")

    original_img = img.copy()

    # -------------------------------------------------
    # 2. Remove illumination bias (important for dermoscopy)
    # -------------------------------------------------
    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge((l,a,b))
    original_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -------------------------------------------------
    # 3. Preprocess EXACTLY like ResNet training
    # -------------------------------------------------
    img = cv2.resize(original_img, (224, 224))
    img = img.astype("float32")
    img = preprocess_input(img)
    img_array = np.expand_dims(img, axis=0)

    # -------------------------------------------------
    # 4. Get CBAM feature map (layer before GAP)
    # -------------------------------------------------
    cbam_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            cbam_layer = model.layers[model.layers.index(layer) - 1].name
            break

    if cbam_layer is None:
        raise ValueError("CBAM feature layer not found")

    # -------------------------------------------------
    # 5. Create GradCAM++ model
    # -------------------------------------------------
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(cbam_layer).output,
            model.output
        ]
    )

    # -------------------------------------------------
    # 6. GradCAM++ computation
    # -------------------------------------------------
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, feature_maps)

    feature_maps = feature_maps[0]
    grads = grads[0]

    grads_power_2 = grads ** 2
    grads_power_3 = grads ** 3

    sum_grads = tf.reduce_sum(feature_maps * grads_power_3, axis=(0,1))

    eps = 1e-8
    alpha = grads_power_2 / (2 * grads_power_2 + sum_grads + eps)

    positive_gradients = tf.nn.relu(grads)
    weights = tf.reduce_sum(alpha * positive_gradients, axis=(0,1))

    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # -------------------------------------------------
    # 7. Remove weak activations (important)
    # -------------------------------------------------
    heatmap[heatmap < 0.35] = 0

    # -------------------------------------------------
    # 8. Resize heatmap to original image
    # -------------------------------------------------
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # -------------------------------------------------
    # 9. Apply colormap
    # -------------------------------------------------
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # -------------------------------------------------
    # 10. Overlay
    # -------------------------------------------------
    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap_color, 0.3, 0)

    # -------------------------------------------------
    # 11. Save
    # -------------------------------------------------
    cv2.imwrite(output_path, superimposed_img)

    return output_path
