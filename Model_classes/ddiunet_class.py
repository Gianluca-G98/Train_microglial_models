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
                        "input_shape": (128, 128, 1),
                        "loss_function": "dice",
                        "optimizer": "adam",
                        "learning_rate": 0.0001,
                        "architecture": "DDI_NoConv1",
                        "dataset": "GBC-2images-EqAdapthist-processed",
                        "n_filters": [32, 64, 128, 256],
                        "earlystopping_patience": 20
                    })
    config = wandb.config  # We'll use this to configure our experiment
    return config

    
def build_class(config):
    model_class = InceptionUnet((None,None,1), filter_list=config.n_filters)
    model_class.build_encoder()
    model_class.build_decoder()
    model_class = model_class.compile_model() 
    model_class.summary()
    return model_class


### DEFINE MODEL CLASS
class InceptionUnet:

    def __init__(self, input_shape, filter_list):
        self.input = tfk.Input(shape=input_shape)
        self.current_encoder_green = self.input[:,:,:,0:1]  # I initialize like this then put the last layer created while going
        self.skipped = []  # Skipped list of microglia
        self.dense_path = []
        self.filter_list = filter_list  # e.g. [32, 64, 128, 256, 512]

    def DilatedConv(self, n_filters, input_block): 
        # Input block is the layer we want to start from for the 4 conv
        layer1 = tfkl.Conv2D(n_filters, 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(input_block)
        layer2 = tfkl.Conv2D(n_filters, 3, dilation_rate=2 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(input_block)
        layer3 = tfkl.Conv2D(n_filters, 3, dilation_rate=4 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(input_block)
        layer4 = tfkl.Conv2D(n_filters, 3, dilation_rate=6 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(input_block)
        return tfkl.concatenate([layer1, layer2, layer3, layer4], axis=3)      

    def EncoderMiniBlock(self, n_filters, dropout_prob=0.3, max_pooling=True):
        self.current_encoder_green = self.DilatedConv(n_filters, input_block=self.current_encoder_green)
        self.current_encoder_green = tfkl.BatchNormalization()(self.current_encoder_green, training=False)

        if dropout_prob > 0:     
            self.current_encoder_green = tfkl.Dropout(dropout_prob)(self.current_encoder_green)

        # Copy feature map before maxpooling:
        self.skipped.append(self.current_encoder_green)

        if max_pooling:
            self.current_encoder_green = tfkl.MaxPooling2D(pool_size = (2,2))(self.current_encoder_green)
        else:
            pass

    def build_encoder(self):
        for idx, n in enumerate(self.filter_list):
            self.EncoderMiniBlock(n_filters=n, dropout_prob=0)

        for idx, skipped_layer in enumerate(self.skipped):
            dense1 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(skipped_layer)
            bnorm1 = tfkl.BatchNormalization()(dense1)
            dense2 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(bnorm1)
            bnorm2 = tfkl.BatchNormalization()(dense2)

            self.dense_path.append(tfkl.add([bnorm1, bnorm2]))

        # Now I'm going to INVERT the list that I will then have to use in the decoder
        self.dense_path.reverse()  # I reverse the features because I concatenate the last ones first
        self.filter_list.reverse()  # I reverse the filter list to pass it to the decoder



    def DecoderMiniBlock(self, n_filters, skipped_layer):
        self.current_decoder = tfkl.Conv2DTranspose(n_filters, (3,3), strides=(2,2),
                                        padding='same')(self.current_decoder)
        self.current_decoder = tfkl.concatenate([self.current_decoder, skipped_layer], axis=3)
        self.current_decoder = self.DilatedConv(n_filters, input_block=self.current_decoder)

    def build_decoder(self):
        # The first layer of the decoder is the sum of the last layers of the encoder (green+blue)
        self.current_decoder = self.current_encoder_green
        for idx, n in enumerate(self.filter_list):
            self.DecoderMiniBlock(n_filters=n, skipped_layer=self.dense_path[idx])
            
        # After finishing decoding I add a final filter to get a single feature map
        self.current_decoder = tfkl.Conv2D(filters=1, kernel_size=1, 
                                           activation="sigmoid", padding="same")(self.current_decoder)

    def compile_model(self):
        return tfk.Model(inputs=self.input, outputs=self.current_decoder)