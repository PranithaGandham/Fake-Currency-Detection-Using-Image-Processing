from flask import Flask, render_template, request
from PIL import Image
from Project import validate
from denomination import extract

app = Flask(__name__,static_folder='static')

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./Deployment/" + imagefile.filename
    imagefile.save(image_path)

    denomination = int(extract(image_path))
    validation = validate(image_path,denomination)

    return validation

if __name__ == "__main__":
    app.run(port=3000, debug=True)