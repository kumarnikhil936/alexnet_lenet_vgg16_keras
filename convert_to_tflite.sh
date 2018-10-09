tflite_convert --keras_model_file=$PWD/simple_model.h5 --output_file=$PWD/simple_model.tflite 
tflite_convert --keras_model_file=$PWD/alexnet.h5 --output_file=$PWD/alexnet.tflite 
tflite_convert --keras_model_file=$PWD/lenet.h5 --output_file=$PWD/lenet.tflite 
