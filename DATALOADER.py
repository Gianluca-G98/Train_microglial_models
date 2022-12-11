
import numpy as np
import os
import wandb
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(train_X, train_y, validation_X, validation_y, config):
    print("\n\n####")
    print("Check train and validation shapes:")
    print(train_X.shape, train_y.shape, validation_X.shape, validation_y.shape)
    print("####\n\n")

    data_gen_args_X = dict(rotation_range=20,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        fill_mode='reflect',
                        horizontal_flip=True, 
                        vertical_flip=True)
    data_gen_args_y = dict(rotation_range=20,
                        width_shift_range=0.01,
                        height_shift_range=0.01,
                        fill_mode='reflect',
                        horizontal_flip=True, 
                        vertical_flip=True,
                        preprocessing_function=np.round)
                        
    train_datagen_X = ImageDataGenerator(**data_gen_args_X)
    train_datagen_y = ImageDataGenerator(**data_gen_args_y)

    image_generator = train_datagen_X.flow(train_X, seed=42, batch_size=config["batch_size"], shuffle=False)
    mask_generator = train_datagen_y.flow(train_y, seed=42, batch_size=config["batch_size"], shuffle=False)

    # combine generators into one which yields image and masks 
    train_generator = zip(image_generator, mask_generator)

    # Create generator to test
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(validation_X, validation_y, 
                                        seed=42, batch_size=config["batch_size"], shuffle=False)
    return train_generator, val_generator


def load_data_raw(split, wandb_config):
    with wandb.init(project=wandb_config["project_artifacts"], entity=wandb_config["entity"], job_type="load-data") as run:
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact(wandb_config["testing_artifact_name"])

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download()

        #split=["green_fusion", "manual_annot"]
        green_fusion, manual_annot = read_raw(raw_dataset, split)
        print("IMPORTED DATA:", green_fusion.shape)
        return green_fusion, manual_annot

def read_raw(data_dir, split):
    filename1 = split[0] + ".np"
    data1 = np.load(os.path.join(data_dir, filename1))

    filename2 = split[1] + ".np"
    data2 = np.load(os.path.join(data_dir, filename2))

    return data1, data2


def load_data_processed(split, wandb_config):
    with wandb.init(project=wandb_config["project_artifacts"], entity=wandb_config["entity"], job_type="load-data") as run:
         
        # ✔️ declare which artifact we'll be using
        Microglia0_preprocess_artifact = run.use_artifact(wandb_config["training_artifact_name"])

        # 📥 if need be, download the artifact
        Microglia0_preprocess = Microglia0_preprocess_artifact.download()

        if split == ["train", "validation", "test"]:
            train_X, train_y, validation_X, validation_y, test_X, test_y = read_processed(Microglia0_preprocess, split)
            return train_X, train_y, validation_X, validation_y, test_X, test_y

        elif split == ["train", "test"]: 
            train_X, train_y, test_X, test_y = read_processed(Microglia0_preprocess, split)
            return train_X, train_y, test_X, test_y
        
        elif split == ["train", "validation"]: 
            train_X, train_y, validation_X, validation_y = read_processed(Microglia0_preprocess, split)
            return train_X.astype("float32"), train_y.astype("float32"),\
                 validation_X.astype("float32"), validation_y.astype("float32")


def read_processed(data_dir, split):
    list_datasets = []  # list where I append all the loaded datasets
    for name in split:
        filename1 = name + "_X" + ".np"  
        data1 = np.load(os.path.join(data_dir, filename1))
        list_datasets.append(data1)

        filename2 = name + "_y" + ".np"  
        data2 = np.load(os.path.join(data_dir, filename2))
        list_datasets.append(data2)

    return tuple(list_datasets)  # tuples are automatically unpacked by python when returned!