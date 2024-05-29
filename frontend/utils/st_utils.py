import base64
import json
import os
from math import floor
from random import sample
from time import sleep
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from PIL.Image import Image as ImageObject
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torchvision import transforms
from utils.cnn import CNN, load_data, load_model_weights


def treat_uploaded_images(upload: List[UploadedFile] or UploadedFile, size=(224, 224)):
    """
    Input argument can be one or more files (Expected types: 'png', 'jpg', 'jpeg')
    """

    images = []
    # Ensure upload is a list even if it's a single UploadedFile
    if not isinstance(upload, list):
        upload = [upload]

    for document in upload:
        try:
            # Read the image file using the buffer interface provided by UploadedFile
            img = Image.open(document).convert("L")  # Convert image to black and white
            img_resized = img.resize(size)
            images.append(img_resized)
        except Exception as e:
            print(f"There was an error when receiving the uploaded image: {e}")
    return images


def save_to_local(files: List[UploadedFile] or UploadedFile):
    for i, document in enumerate(files):
        document.save(f"backend/images/user_uploaded/upload_{i}.png")
    print(f"Saved {len(files)} pictures.")


def choose_random_images(images: List[ImageObject] or ImageObject, number: int):
    """
    Chooses couple images to display to the user.
    """

    try:
        return sample(images, number)
    except Exception as e:
        print(f"There was an error: {e}")

    return sample(images, number)


def set_bg(main_bg):
    """
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    """
    # set bg name
    main_bg_ext = "jpg"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover
        }}
        .frost {{
            width: 400px;
            height: 200px;
            position: absolute;
            background: inherit;
        }}
        .frost:before {{
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            filter: blur(12px);
            background: inherit;
        }}
        .content {{
            position: absolute;
            width: 340px;
            height: 140px;
            top: 30px;
            left: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def image_demo_display(images):
    # Image displaying
    if images is not None and len(images) != 0:
        if floor(len(images) / 2) < 4:
            img_toshow = choose_random_images(images, floor(len(images) / 2))
        else:
            img_toshow = choose_random_images(images, 5)

        return img_toshow


def display_images_demo(image):
    try:
        st.image(image, caption="Displayed Image", use_column_width=True)
    except Exception as e:
        print(f"There was an error: {e.__class__.__name__}")


def fake_wait(time: int, text: str):
    with st.spinner(text=text):
        sleep(time)
    return 1


def load_images(uploaded_files):
    images = []
    for file in uploaded_files:
        # Assuming treat_uploaded_images is a function that processes each uploaded file
        image = treat_uploaded_images(file)
        images.append(image)
    return images


def predict_image_clasification(images: List[ImageObject]):
    # Load data and model

    # Display the image with the name of the label
    classnames = [
        "Bedroom",
        "Coast",
        "Forest",
        "Highway",
        "Industrial",
        "Inside city",
        "Kitchen",
        "Living room",
        "Mountain",
        "Office",
        "Open country",
        "Store",
        "Street",
        "Suburb",
        "Tall building",
    ]

    # Load model
    model_weights = load_model_weights("resnet50-5epochV3")
    my_trained_model = CNN(
        torchvision.models.resnet50(weights="DEFAULT"), len(classnames)
    )
    my_trained_model.load_state_dict(load_model_weights("resnet50-5epochV3"))
    for image in images:
        predicted_label = my_trained_model.predict_single_image(image)

        return classnames[predicted_label]
