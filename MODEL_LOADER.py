### Import libs 
import importlib
import json
from keras_unet_collection.losses import tversky, dice

with open('w&b_secret_key.json', 'r') as file_to_read:
    json_data = json.load(file_to_read)
    architecture = json_data["architecture"]  # take the architecture name from the json

# Now we import all the function from inside the script indicated inside the "architecture" variable 
segmentation_class = importlib.import_module(architecture)
globals().update(segmentation_class.__dict__)  # we need to update the globals with the modules insede the variable


### Define function returning the built model
def build_model(config):
    model = build_class(config=config)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]), 
              loss=dice, metrics=["BinaryAccuracy", "Recall", "Precision"])
    return model