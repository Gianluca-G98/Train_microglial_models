### Load libraries and scripts
import json
import math
import random
import tensorflow as tf 
import numpy as np
from wandb.keras import WandbCallback
from DATALOADER import *

### Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


### Login on wandb
with open('w&b_secret_key.json', 'r') as file_to_read:
    wandb_config_dict = json.load(file_to_read) # copy the json params as dict

wandb.login(key=wandb_config_dict["w&b_key"])
print("The model currently tested is: ", wandb_config_dict["architecture"])


from MODEL_LOADER import *
from test_models import *

### Set model name
MODEL_NAME = wandb_config_dict["architecture"][14:-6]
print("MODEL_NAME:", MODEL_NAME)


### Load raw and processed data
green_fusion, y_true = load_data_raw(split=["green_blue_fusion2", "annot2"], wandb_config=wandb_config_dict)
print("Import raw data completed")
train_X, train_y, validation_X,\
     validation_y = load_data_processed(split=["train", "validation"], wandb_config=wandb_config_dict)
print("Import processed data completed")


### Init wandb run and return 
config = build_wandb_config(wandb_config_dict)


### Augment data
train_generator, val_generator = augment_data(train_X, train_y, 
                                            validation_X, validation_y, config)

steps_per_epoch = np.ceil(np.shape(train_X)[0]/config.batch_size) 
steps_per_epoch_val = np.ceil(np.shape(validation_X)[0]/config.batch_size) 

### Define callbacks
wandb_callback = WandbCallback(monitor='val_loss',  
                               log_weights=False,
                               log_evaluation=False,
                               validation_steps=steps_per_epoch_val, 
                               save_model=False
                               )

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=config['earlystopping_patience'], 
    verbose=0, mode='auto',
    restore_best_weights=True
)

RLPl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            factor=0.1, patience=10,
                                             mode='auto', min_lr=0)

callbacks = [earlystopper, wandb_callback, RLPl]                                        


### Build model
model = build_model(config=config)

# Disable untraced functions warnings
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

### Start training
history = model.fit(train_generator, 
                    epochs=config.epochs, 
                    batch_size=config.batch_size, 
                    callbacks=callbacks, 
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_generator)

best_epoch_idx = np.argmin(history.history["val_loss"])
precision_on_best_epoch = history.history["val_precision"][best_epoch_idx]
recall_on_best_epoch = history.history["val_recall"][best_epoch_idx]
f1_on_best_epoch = np.round(2*((precision_on_best_epoch*recall_on_best_epoch)/(precision_on_best_epoch+recall_on_best_epoch)), 4)
wandb.log({"best_precision": precision_on_best_epoch,
           "best_recall": recall_on_best_epoch,
           "best_f1": f1_on_best_epoch})

# Predict test image (green_fusion)
y_pred = np.round(predict_test(model=model, green_fusion=green_fusion[:,:,0:2], y_true=y_true, size_vert=256, size_oriz=256))

# Create 4-colors plot
img_path = 'Predicted_images/'
img_name = MODEL_NAME + '.png'
y_plot = colored_plot(y_true, y_pred, img_name=img_name, img_path=img_path)

# I calculate test metrics and log them to wandb
idx_test = math.ceil((y_plot.shape[0]//2) / 128)*128
log_test_metrics(y_true, y_pred, idx_test=idx_test)

wandb.finish()