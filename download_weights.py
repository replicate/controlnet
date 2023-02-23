import argparse

# add command line arg for model type
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="canny", help="Model type to download")
# add a binary flag to wipe the weights folder
parser.add_argument("--wipe", action="store_true", help="Wipe the weights folder")
args = parser.parse_args()

MODEL_TYPE = args.model_type

from utils import model_dl_urls, annotator_dl_urls, download_model

for model_name in annotator_dl_urls.keys():
    download_model(model_name, annotator_dl_urls)

if MODEL_TYPE != "all":
    download_model(MODEL_TYPE, model_dl_urls)
else:
    for model_name in model_dl_urls.keys():
        download_model(model_name, model_dl_urls)