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
                     "architecture": "MyDualChannel_respath",
                     "dataset": "GBC-2images-EqAdapthist-processed",
                     "n_filters": [32, 64, 128, 256],
                     "earlystopping_patience": 50,
                     "dropout_prob": 0, 
                     "kernel_initializer": "HeNormal",
                     "activation_function": "relu",
                     "output_activation":"hard_sigmoid",
                 })
    config = wandb.config  # We'll use this to configure our experiment
    return config

    
def build_class(config):
    model_class = DualChannel((None,None,1), filter_list=[32, 64, 128, 256],
                            dropout_prob=config.dropout_prob, 
                            kernel_initializer=config.kernel_initializer,
                    activation_function=config.activation_function,
                    output_activation=config.output_activation)
    model_class.build_encoder()
    model_class.build_decoder()
    model_class = model_class.compile_model()
    model_class.summary()
    return model_class


### DEFINE MODEL CLASS
class DualChannel:

    def __init__(self, input_shape, filter_list, dropout_prob, kernel_initializer, activation_function, output_activation):
        self.input = tfk.Input(shape=input_shape)
        self.current_encoder_green = self.input[:,:,:,0:1]  # I initialize like this then put the last layer created while going
        self.skipped = []  # Skipped connections for microglia
        self.res_path = []
        self.filter_list = filter_list  # e.g. [32, 64, 128, 256, 512]
        self.dropout_prob = dropout_prob
        self.kernel_initializer = kernel_initializer
        self.activation_function = activation_function
        self.output_activation = output_activation

    def DualChannelBlock(self, n_filters, input_block):
        channel_1 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(input_block)           
        channel_1_2 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(channel_1)                      
        channel_1_3 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(channel_1_2)
        concat_ch1 = tfkl.concatenate([channel_1, channel_1_2, channel_1_3], axis=3)

        channel_2 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(input_block)           
        channel_2_2 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(channel_2)                      
        channel_2_3 = tfkl.Conv2D(n_filters, 3, activation=self.activation_function, padding='same',
                    kernel_initializer=self.kernel_initializer)(channel_2_2)
        concat_ch2 = tfkl.concatenate([channel_2, channel_2_2, channel_2_3], axis=3)
        
        return tfkl.Add()([concat_ch1, concat_ch2])

    def EncoderMiniBlock(self, n_filters, max_pooling=True):
        self.current_encoder_green = self.DualChannelBlock(n_filters, input_block=self.current_encoder_green)

        if self.dropout_prob > 0:     
            self.current_encoder_green = tfkl.Dropout(self.dropout_prob)(self.current_encoder_green)

        # Copy feature map before maxpooling:
        self.skipped.append(self.current_encoder_green)

        if max_pooling:
            self.current_encoder_green = tfkl.MaxPooling2D(pool_size = (2,2))(self.current_encoder_green) 
        else:
            pass

    def build_encoder(self):
        for idx, n in enumerate(self.filter_list):
            self.EncoderMiniBlock(n_filters=n)
        
        for idx, skipped_layer in enumerate(self.skipped):
            dense1 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(skipped_layer)
            skip_con1 = tfkl.Conv2D(self.filter_list[idx], 1, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(skipped_layer)
            added_1 = tfkl.add([dense1, skip_con1])
            
            dense2 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_1)
            skip_con2 = tfkl.Conv2D(self.filter_list[idx], 1, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_1)
            added_2 = tfkl.add([dense2, skip_con2])
            
            dense3 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_2)
            skip_con3 = tfkl.Conv2D(self.filter_list[idx], 1, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_2)
            added_3 = tfkl.add([dense3, skip_con3])
            
            dense4 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_3)
            skip_con4 = tfkl.Conv2D(self.filter_list[idx], 1, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_3)
            added_4 = tfkl.add([dense4, skip_con4])
            
            dense5 = tfkl.Conv2D(self.filter_list[idx], 3, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_4)
            skip_con5 = tfkl.Conv2D(self.filter_list[idx], 1, dilation_rate=1 , activation="relu", padding='same',
                    kernel_initializer='HeNormal')(added_4)
            added_5 = tfkl.add([dense5, skip_con5])

            self.res_path.append(added_5)

        # Now I'm going to INVERT the list that I will then have to use in the decoder
        self.res_path.reverse()  # I reverse the features because I concatenate the last ones first
        self.skipped.reverse()  # I don't need this variable anymore, now I use res_path
        self.filter_list.reverse()  # I reverse the filter list to pass it to the decoder
        

    def DecoderMiniBlock(self, n_filters, skipped_layer):
        self.current_decoder = tfkl.Conv2DTranspose(n_filters, (3,3), strides=(2,2),
                                        padding='same')(self.current_decoder)
        self.current_decoder = tfkl.concatenate([self.current_decoder, skipped_layer], axis=3)
        self.current_decoder = self.DualChannelBlock(n_filters, input_block=self.current_decoder)

    def build_decoder(self):
        # The first layer of the decoder is the sum of the last layers of the encoder (green+blue)
        self.current_decoder = self.current_encoder_green
        
        for idx, n in enumerate(self.filter_list):
            self.DecoderMiniBlock(n_filters=n, skipped_layer=self.res_path[idx]) 
            
        # Also finished decoding I add a final filter to get a single feature map
        self.current_decoder = tfkl.Conv2D(filters=1, kernel_size=1, 
                                           activation=self.output_activation, padding="same")(self.current_decoder)

    def compile_model(self):
        return tfk.Model(inputs=self.input, outputs=self.current_decoder)