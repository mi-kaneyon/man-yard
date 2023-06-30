import os
from PIL import Image, ImageResampling

def resize_images(directory):
    count = 0

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                width, height = img.size
                new_width = 512
                new_height = int(new_width * height / width)
                resized_img = img.resize((new_width, new_height), ImageResampling.LANCZOS)
                resized_img.save(filepath)
                count += 1
            except Exception as e:
                print(f"Error resizing {filename}: {e}")

    print(f"Resized {count} files successfully.")

# Specify the directory containing the images
directory = "./images"

# Call the resize_images function
resize_images(directory)
