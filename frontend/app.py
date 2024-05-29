from time import sleep

import streamlit as st
from utils.st_utils import (
    display_images_demo,
    fake_wait,
    image_demo_display,
    predict_image_clasification,
    save_to_local,
    set_bg,
    treat_uploaded_images,
)


def main():

    if "analyse_ready" not in st.session_state:
        st.session_state["analyse_ready"] = False

    # Set page configuration to use the full width of the screen and add a page icon
    st.set_page_config(
        layout="centered", page_icon="🏠", page_title="House Image Classifier"
    )

    # Set a dynamic background image based on the time of day

    # Set the title and description of the application with enhanced formatting
    st.markdown(
        "<h1 style='text-align: center; color: black;'>🏡 House Image Classifier</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: justify;">
            Welcome to the demo version of our image classification tool. 
            Please take your time to explore the interface. You can upload your images using the sidebar on the left.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for image upload with additional instructions and a progress bar
    with st.sidebar:
        st.image("frontend/images/logo.png", width=200)
        st.title("🖼️ Upload Images")

        uploader = st.file_uploader(
            label="Drag and drop your images here",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload your images in PNG, JPG, or JPEG format. ZIP files are not supported.",
        )
        if uploader:
            st.success("Images have been uploaded successfully!")
        # Attempt to process the uploaded images with preprocessing
        images = []
        if uploader:
            try:
                images = treat_uploaded_images(uploader)
                st.session_state.analyse_ready = True
            except Exception as e:
                st.error(f"An error occurred while processing the images: {e}")

    # Display the number of uploaded images with a custom message
    if images:
        st.success(f"You have uploaded {len(images)} images.")
    else:
        st.info("Awaiting image uploads...")

    if images:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("")
        with col2:
            st.image(
                images,
                caption="Uploaded Images",
                width=250,
                use_column_width="auto",
                output_format="auto",
            )
        with col3:
            st.markdown("")

        # Button to start image analysis with a confirmation dialog
        if st.session_state.analyse_ready:
            st.markdown(
                "🔍 If the images are correct, click on **Analyse Images** to start classification."
            )
            col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("")
        with col2:
            if st.button(label="🔬 Analyse Images"):
                with st.spinner("Analysis in progress..."):
                    # Simulate the analysis process with fake wait times and progress bars
                    sleep(1)
                    # Reset the session state to allow for new uploads
                    st.session_state.analyse_ready = False

                    # Display the classification results with a placeholder function
                    answer = predict_image_clasification(
                        images
                    )  # Placeholder for new feature
                    emoji_dict = {
                        "Bedroom": "🛏️",
                        "Coast": "🏖️",
                        "Forest": "🌲",
                        "Highway": "🛣️",
                        "Industrial": "🏭",
                        "Inside city": "🏙️",
                        "Kitchen": "🍳",
                        "Living room": "🛋️",
                        "Mountain": "⛰️",
                        "Office": "🏢",
                        "Open country": "🏞️",
                        "Store": "🏬",
                        "Street": "🛤️",
                        "Suburb": "🏘️",
                        "Tall building": "🏬",
                    }
                    if answer:
                        st.markdown(
                            f"<h1 style='text-align: center; color: black;'> {answer} {emoji_dict[answer]}</h1>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error("Unable to classify the image. Please try again.")
        with col3:
            st.markdown("")


if __name__ == "__main__":
    main()
