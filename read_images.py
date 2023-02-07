import numpy as np

from PIL import Image


def load_npz(path):
    with np.load(path) as data:
        img = data['arr_0']
    return img

def main():
    PATH = '/tmp/openai-2023-02-07-08-44-10-391655/samples_64x64x64x3.npz'
    images = load_npz(PATH)
    for c, img in enumerate(images, 0):
        pil_img = Image.fromarray(img)
        pil_img.save(f'images/sample_img_{c}.jpg')
    return

if __name__ == "__main__":
    main()
