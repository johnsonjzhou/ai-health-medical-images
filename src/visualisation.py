import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
import numpy as np
from typing import List


def plot_images(
    dataset: tf.data.Dataset,
    class_names: List[str]
):
    fig = plt.figure(figsize=(6, 3), dpi=192)
    gs = GridSpec(nrows=2, ncols=4)
    for images, labels in dataset.take(1):
        for i in range(8):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.imshow(X=images[i], cmap="gray")
            ax.set_title(class_names[labels[i]], fontsize="small")
            ax.axis("off")
    plt.show()
    return


def plot_predictions(
    dataset: tf.data.Dataset,
    class_names: List[str],
    model: tf.keras.Model
):
    fig = plt.figure(figsize=(6, 3), dpi=192)
    gs = GridSpec(nrows=2, ncols=4, hspace=0.5)

    for images, labels in dataset.take(1):
        for i in range(8):
            image = images[i]
            label = labels[i]
            predicted_probabilities = model.predict(image[tf.newaxis,])[0]
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.imshow(X=image, cmap="gray")
            ax.set_title(
                label=(
                    f"True: {class_names[label]}"
                    f"\nPredicted: Normal {predicted_probabilities[0]:.2%}"
                    f"\nPredicted: Stroke {predicted_probabilities[1]:.2%}"
                ),
                fontsize="xx-small"
            )
            ax.axis("off")
    plt.show()
    return


def plot_feature_maps(
    feature_maps,
    model: tf.keras.Model,
    layer: int,
    image_number: int
):
    feature_map = feature_maps[layer]
    n_feature_maps = feature_map.shape[3]
    n_rows = n_feature_maps // 8
    fig_height = (n_rows // 4) * 3
    fig = plt.figure(figsize=(6, fig_height), dpi=192)
    gs = GridSpec(nrows=n_rows, ncols=8, hspace=0.05, wspace=0.05)

    for i in range(n_feature_maps):
        ax = fig.add_subplot(gs[i // 8, i % 8])
        ax.imshow(X=feature_map[0, :, :, i], cmap="viridis")
        ax.axis("off")

    plt.suptitle(
        t=(
            f"Feature map visualisation:"
            f" Image number: {image_number}"
            f", Layer: {model.layers[layer].name}"
            f", Filters: {n_feature_maps}"
        ),
        y=0.92,
        fontsize="xx-small"
    )
    plt.show()
    return


def display_gradcam(
    image: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: str,
    ax: matplotlib.axes.Axes,
    alpha: float = 0.4,
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-1].output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image[tf.newaxis,])
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorise heatmap
    cmap = matplotlib.colormaps["jet"]
    heatmap_colors = cmap(np.arange(256))[:, :3]
    heatmap = heatmap_colors[heatmap]

    # # Create an image with RGB colorized heatmap
    heatmap = tf.keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((image.shape[1], image.shape[0]))
    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
    heatmap = tf.keras.constraints.MinMaxNorm(0., 1., axis=-1)(heatmap)

    # Display the result
    ax.imshow(X=image, cmap="gray")
    ax.imshow(X=heatmap, alpha=alpha)

    return ax


def plot_gradcam(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    class_names: List[str]
):
    fig = plt.figure(figsize=(6, 3), dpi=192)
    gs = GridSpec(nrows=2, ncols=4, hspace=0.5)

    for images, labels in dataset.take(1):
        for i in range(8):
            image = images[i]
            label = labels[i]
            predicted_probabilities = model.predict(image[tf.newaxis,])[0]
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax = display_gradcam(
                image=image,
                model=model,
                last_conv_layer_name="conv2d_1",
                ax=ax,
                alpha=0.8
            )
            ax.set_title(
                label=(
                    f"True: {class_names[label]}"
                    f"\nPredicted: Normal {predicted_probabilities[0]:.2%}"
                    f"\nPredicted: Stroke {predicted_probabilities[1]:.2%}"
                ),
                fontsize="xx-small"
            )
            ax.axis("off")
    plt.show()
    return