import os

BASE_PATH = "D:\\Deep_Learning_Projects\\datasets\\Caltech_101"
IMAGES_PATH = os.path.join(BASE_PATH, "Images")
ANNOTS_PATH = os.path.join(BASE_PATH, "Annotations")

BASE_OUTPUT = "output\\experiment3"

MODEL_PATH = os.path.join(BASE_OUTPUT, "detector")
LB_PATH = os.path.join(BASE_OUTPUT, "lb.pickle")
PLOTS_PATH = os.path.join(BASE_OUTPUT, "plots")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

#ImageNet mean and standard deviation for the pretrained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

INIT_LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
VALID_LABELS = ["Airplane", "Face", "Motorcycle", "Leopards", "stop_sign", "watch", "bonsai",
                "car_side", "chandelier", "grand_piano", "hawksbill", "ketch", "accordion",
                "anchor", "ant", "barrel", "bass", "beaver", "binocular", "ceiling_fan",
                "brain", "buddha", "butterfly", "camera", "cannon", "cellphone", "chair",
                "cougar_body", "crab", "crocodile", "cup", "dalmatian", "dolphin", "dragonfly",
                "elephant", "emu", "flamingo", "garfield", "gramophone", "headphone",
                "hedgehog", "helicopter", "kangaroo", "lamp", "laptop", "llama",
                "lobster", "lotus", "mandolin", "pizza", "platypus", "revolver",
                "rhino", "rooster", "saxophone", "scorpion", "scrissors", "sea_horse",
                "soccer_ball", "starfish", "strawberry", "sunflower", "umbrella",
                "weelchair", "wild_cat", "windsor_chair", "wrench"]
