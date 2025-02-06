set NUMBER_OF_CLASSES=2
set PATH_TO_DATASET=.\maritime-flags-dataset\balanced_two_flags

py classify.py --cnn_type alexnet --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type vgg16 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type vgg19 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type inceptionv3 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type resnet-18 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type resnet-34 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type resnet-50 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type resnext --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type mobilenetv2 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type efficientnet-b0 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type efficientnet-b1 --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type cspnet --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001
py classify.py --cnn_type convnext --n_classes %NUMBER_OF_CLASSES% --path_to_dataset %PATH_TO_DATASET% --batch_size 32 --n_epochs 10 --learning_rate 0.001