### IMPORT LIBS
import tensorflow as tf 
import wandb

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

def build_wandb_config(wandb_config_dict):
    # Init wandb run
    run = wandb.init(project=wandb_config_dict["project_training"], entity=wandb_config_dict["entity"],
                    config={  # and include hyperparameters and metadata
                        "epochs": 200, #
                        "batch_size": 64, #
                        "input_shape": (128, 128, 2),
                        "loss_function": "dice",
                        "optimizer": "adam",
                        "learning_rate": 0.0001,
                        "architecture": "FuseSeg",
                        "dataset": "GBC-2images-EqAdapthist-processed",
                        "n_filters": [32, 64, 128, 256, 512],
                        "earlystopping_patience": 50,
                        "dropout_prob": 0, 
                        "kernel_initializer": "HeNormal",
                        "activation_function": "relu",
                        "output_activation":"sigmoid",
                    })

    config = wandb.config  # We'll use this to configure our experiment
    return config

    
def build_class(config):
    input_layer = tf.keras.Input(shape=(None,None,2))
    input_green = input_layer[:,:,:,0:1]
    input_blue = input_layer[:,:,:,1:2]

    n_filters = config.n_filters

    # Create Encoder Green
    block1_green, skip1_green = EncoderMiniBlock(input_green, n_filters[0])
    block2_green, skip2_green = EncoderMiniBlock(block1_green, n_filters[1])
    block3_green, skip3_green = EncoderMiniBlock(block2_green, n_filters[2])
    block4_green, skip4_green = EncoderMiniBlock(block3_green, n_filters[3])
    block5_green, skip5_green = EncoderMiniBlock(block4_green, n_filters[4])

    # Create Encoder Blue
    block1_blue, skip1_blue = EncoderMiniBlock(input_blue, n_filters[0])
    block2_blue, skip2_blue = EncoderMiniBlock(block1_blue, n_filters[1])
    block3_blue, skip3_blue = EncoderMiniBlock(block2_blue, n_filters[2])
    block4_blue, skip4_blue = EncoderMiniBlock(block3_blue, n_filters[3])
    block5_blue, skip5_blue = EncoderMiniBlock(block4_blue, n_filters[4])

    # Sum skipped connection
    fuse1 = tfkl.add([skip1_green, skip1_blue])
    fuse2 = tfkl.add([skip2_green, skip2_blue])
    fuse3 = tfkl.add([skip3_green, skip3_blue])
    fuse4 = tfkl.add([skip4_green, skip4_blue])
    fuse5 = tfkl.add([skip5_green, skip5_blue])

    last_encoder = tfkl.add([block5_green, block5_blue])

    # Create Decoder
    up1 = DecoderMiniBlock(last_encoder, fuse5, n_filters[4])
    up2 = DecoderMiniBlock(up1, fuse4, n_filters[3])
    up3 = DecoderMiniBlock(up2, fuse3, n_filters[2])
    up4 = DecoderMiniBlock(up3, fuse2, n_filters[1])
    up5 = DecoderMiniBlock(up4, fuse1, n_filters[0])

    outputs = tfkl.Conv2D(filters=1, kernel_size=(1,1), activation="sigmoid")(up5)
    model_class = tf.keras.Model(inputs=input_layer, outputs=outputs)

    model_class.summary()

    return model_class


### DEFINE MODEL FUNCTIONS
def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = tfkl.Conv2D(n_filters, 
                  3,  # filter size
                  activation="relu",
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = tfkl.Conv2D(n_filters, 
                  3,  # filter size
                  activation="relu",
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
  
    conv = tfkl.BatchNormalization()(conv, training=False)
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    skip_connection = conv    
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = tfkl.Conv2DTranspose(
                 n_filters,
                 (3,3),
                 strides=(2,2),
                 padding='same')(prev_layer_input)
    merge = tfkl.concatenate([up, skip_layer_input], axis=3)
    conv = tfkl.Conv2D(n_filters, 
                 3,  
                 activation="relu",
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = tfkl.Conv2D(n_filters,
                 3, 
                 activation="relu",
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv