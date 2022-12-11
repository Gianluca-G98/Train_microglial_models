import numpy as np 
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

### Function to create 4-colors plot
def colored_plot(y_true, y_pred, img_name, img_path):
    y_plot = np.zeros_like(y_true, dtype=np.float32)-1
    print(np.unique(y_plot))

    y_pred_rounded = np.round(y_pred)
    y_plot[np.logical_and(np.logical_not(y_true.astype(bool)), np.logical_not(y_pred_rounded.astype(bool)))] = 1  # true negative
    y_plot[np.logical_and(y_true.astype(bool), y_pred_rounded.astype(bool))] = 2  # true positive
    y_plot[np.logical_and(np.logical_not(y_true.astype(bool)), y_pred_rounded.astype(bool))] = 3  # false positive
    y_plot[np.logical_and(y_true.astype(bool), np.logical_not(y_pred_rounded.astype(bool)))] = 4  # false negative
    print(np.unique(y_plot))

    colors = ['blue', 'green', 'red', 'yellow']
    bounds = [1,2,3,4,5]

    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(y_plot, interpolation='none', cmap=cmap, norm=norm)
    path = img_path + img_name
    plt.imsave(fname=path, arr=y_plot, cmap=cmap, format='png')
    return y_plot


### Log only test region score (originally second half of img.2)
def log_test_metrics(y_true, y_pred, idx_test):
    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()
    image_precision = np.round(tf.keras.backend.get_value(precision_metric(y_true[idx_test:, :], np.round(y_pred[idx_test:, :]))), 4)
    image_recall = np.round(tf.keras.backend.get_value(recall_metric(y_true[idx_test:, :], np.round(y_pred[idx_test:, :]))), 4)
    image_f_score = np.round(2*((image_precision*image_recall)/(image_precision+image_recall)), 4)
    wandb.log({"Image Recall": image_recall,
              "Image Precision": image_precision,
              "Image F1-score": image_f_score})


### Predict the second image (green_fusion)
def predict_test(model, green_fusion, y_true, size_vert, size_oriz):
    y_pred = np.zeros_like(y_true, dtype=np.float32)  # Create a ndarray for the prediction
    size_vert = size_vert
    size_oriz = size_oriz
    i = 0
    for vert in range(0, y_true.shape[0], size_vert): # iter vertically
        for oriz in range(0, y_true.shape[1], size_oriz): # iter horizontally
            input_tile = green_fusion[vert:vert+size_vert, oriz:oriz+size_oriz, :]  # take corresponding green_fusion region
            input_tile = tf.expand_dims(input_tile, axis=0)  # add batch dim
            tile_predicted = model(input_tile, training=False)[0, :,:,0]
            y_pred[vert:vert+size_vert, oriz:oriz+size_oriz] = tile_predicted  # put prediction in the ndarray
            i += 1
    return y_pred