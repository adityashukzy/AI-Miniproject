from PIL import Image
from utils import load_model, postprocess_image, preprocess_image, run_inference

if __name__ == "__main__":
    model = load_model('recolor.h5')
    img = Image.open('uploads/imgg.jpeg')
    img = preprocess_image(img)
    run_inference(model, img)
    img = postprocess_image(img)
    img.show()
