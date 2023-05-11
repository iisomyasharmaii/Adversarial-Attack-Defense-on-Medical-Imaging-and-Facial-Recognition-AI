from flask import Flask, render_template, request, url_for
#import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
#from keras.preprocessing import image
import keras.utils as image
import cv2
import os
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'UPLOAD_FOLDER'
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Load the trained model
load_medical_model = load_model('chest_x_ray_model') # Load model
class_names=['COVID19', 'NORMAL', 'PNEUMONIA']

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

# Define a route to render the HTML webpage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictions')
def index1():
    return render_template('index1.html')

@app.route('/attack')
def index2():
    return render_template('index2.html')

@app.route('/defense')
def index3():
    return render_template('index3.html')

def load_and_prep_image(filename, img_shape=224, scale=False):

  # Read in the image
  img = tf.io.read_file(filename)
  img = tf.io.decode_image(img)
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

# Define a route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    filename = secure_filename(img_file.filename)
    temp_path = os.path.join("static", filename)
    img_file.save(temp_path)
    
    img = load_and_prep_image(temp_path)
    
    prob=load_medical_model.predict(tf.expand_dims(img,axis=0))
    pred_acc=prob.max()*100
    pred_class=class_names[prob.argmax()]

    input_file_url=url_for('static', filename=filename)

    return render_template('results_pred.html', predicted_class=pred_class,predicted_accuracy=pred_acc,image_input_file_name=input_file_url)


#the fast gradient sign method (FGSM) Untargeted method
def adversarial_pattern_FGSM(image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = load_medical_model(tf.expand_dims(image,axis=0))
        loss = tf.keras.losses.MSE(label, prediction)
    
    gradient = tape.gradient(loss, image)
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad

def perturbations_FGSM_generator(image,model):
    image_probs=model.predict(tf.expand_dims(image,axis=0))
    image_pred_index=image_probs.argmax()
    label = tf.one_hot(image_pred_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    return adversarial_pattern_FGSM(image, label).numpy()



# Define a route to handle the form submission
@app.route('/predictFGSM', methods=['POST'])
def predictFGSM():
    img_file = request.files['image']
    filename = secure_filename(img_file.filename)
    temp_path = os.path.join("static", filename)
    img_file.save(temp_path)

    img = load_and_prep_image(temp_path)

    perbutations=perturbations_FGSM_generator(img,load_medical_model)
    adversarial_image = img + perbutations *2
   
    prob=load_medical_model.predict(tf.expand_dims(adversarial_image,axis=0))
    pred_acc=prob.max()*100
    pred_class=class_names[prob.argmax()]
    print(os.listdir())
    print('Function 2 FGSM')
    tf.keras.preprocessing.image.save_img('static/adversarial_image.jpg',adversarial_image)
    
    input_file_url=url_for('static', filename=filename)
    output_file_url=url_for('static', filename='adversarial_image.jpg')
    return render_template('results.html', predicted_class=pred_class,predicted_accuracy=pred_acc,image_input_file_name=input_file_url,image_output_file_name=output_file_url,input_label="Original Image",output_label="Adversarial Attacked Image")    





def perform_nlm_denoising(img):
    """
    return NLM image
    """
    nlm = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    return np.clip(nlm, 0, 255).astype(np.uint8)

# Define a route to handle the form submission
@app.route('/predictNLM', methods=['POST'])
def predictNLM():
    img_file = request.files['image']
    filename = secure_filename(img_file.filename)
    temp_path = os.path.join("static", filename)
    img_file.save(temp_path)
    print(temp_path)
    # Load the noisy image
    img = cv2.imread(temp_path)
    denoised = perform_nlm_denoising(img)

    tf.keras.preprocessing.image.save_img('static/denoised_image.jpg',denoised)

    denoise_image=load_and_prep_image('static/denoised_image.jpg')


    prob=load_medical_model.predict(tf.expand_dims(denoise_image,axis=0))
    pred_acc=prob.max()*100
    pred_class=class_names[prob.argmax()]
    print('Function NLM Denoising')

    input_file_url=url_for('static', filename=filename)
    output_file_url=url_for('static', filename='denoised_image.jpg')
  
    
    return render_template('results.html', predicted_class=pred_class,predicted_accuracy=pred_acc,image_input_file_name=input_file_url,image_output_file_name=output_file_url,input_label="Adversarial Attacked Image",output_label="Adversarial Defense Image")    

if __name__ == '__main__':
    app.run(debug=False)
