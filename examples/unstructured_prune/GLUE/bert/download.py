import wget
import os

print("Downloading dataset...")

# The URL for the dataset zip file.
url = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"

# Download the file (if we haven't already)
if not os.path.exists("./cola_public_1.1.zip"):
    wget.download(url, "./cola_public_1.1.zip")
