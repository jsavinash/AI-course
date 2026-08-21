import importlib

PACKAGE = 'gan_image_generation'

def test_model_importable():
    importlib.import_module(f"{PACKAGE}.model")

def test_data_importable():
    importlib.import_module(f"{PACKAGE}.data")

def test_train_importable():
    importlib.import_module(f"{PACKAGE}.train")

def test_api_importable():
    importlib.import_module(f"{PACKAGE}.api")

def test_package_importable():
    importlib.import_module(PACKAGE)
