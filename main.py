from flask import Flask,jsonify,request
import numpy as np
from tensorflow import keras

app = Flask(__name__)

try:
    model = keras.models.load_model('model/heart_attack_predic_model.h5')
except Exception as e:
    print(e)

@app.route('/heart-attack/predictions',methods=['POST'])
def heart_attack_predictions():
    try:
        input_dict = request.json
        # process data
        val_list = list(input_dict.values())

        # convert that to numpy array of shape(1,13)
        input_data = np.array([val_list])
        predicts = model(input_data)
        predicts = np.multiply(predicts.numpy()[0][0],100)
        return jsonify({
            'msg': 'success',
            'isError': False,
            'data': str(predicts),
            'error': ''
        })
    except Exception as e:
        print(e)
        return jsonify({
            'msg': 'error',
            'isError': True,
            'data': '',
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)