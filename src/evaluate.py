# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from sklearn.metrics import classification_report, confusion_matrix
# from get_data import get_data, read_params
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import os
# import argparse
# import shutil
# import matplotlib.pyplot as plt

# def m_evaluate(config_file):
#     config = get_data(config_file)
#     batch = config['img_augment']['batch_size']
#     class_mode = config['img_augment']['class_mode']
#     test_path = config['model']['test_path']
#     model = load_model('saved_models/trained.h5')
#     config = get_data(config_file)

#     test_gen = ImageDataGenerator(rescale=1./255)
#     test_set = test_gen.flow_from_directory(test_gen, target_size=(255,255), batch_size=batch, class_mode=class_mode) 

#     label_map = (test_set.class_indices)
#     #print(label_map)

#     y_pred = model.predict(test_set)
#     y_pred = np.argmax(y_pred, axis=1)

#     print("Confusion Matrix")
#     sns.heatmap(confusion_matrix(test_set.classes, y_pred))
#     plt.xlabel("Actual Values")
#     plt.ylabel("Predicted Value")
#     plt.savefig('reports/confusion_matrix.png')
#     #plt.show()

#     print("Classification Report")
#     target_names = ['Bulbasaur', 'Charmander', 'Squirtle', 'Tauros']
#     df = pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True))
#     df['support'] = df.support.apply(int)
#     df.style.background_gradient(cmap='viridis', subset=pd.IndexSlice['0':'9', :'f1-score'])
#     df.to_csv('report/classification_report.csv')
#     print('Classification Report and Confusion Matrix Report are saved in reports folder of Template')

# if __name__ == '__main__':
#     args=argparse.ArgumentParser()
#     args.add_argument('--config',default='params.yaml')
#     passed_args=args.parse_args()
#     m_evaluate(config_file=passed_args.config)

from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import argparse
from get_data import get_data
from keras_preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def m_evaluate(config_file):
    config = get_data(config_file)
    batch = config['img_augment']['batch_size']
    class_mode = config['img_augment']['class_mode']
    te_set = config['model']['test_path']
    model = load_model('mlops_project/models/trained.h5')
    config = get_data(config_file)

    test_gen = ImageDataGenerator(rescale = 1./255)
    test_set = test_gen.flow_from_directory(te_set,
                                                target_size = (225,225),
                                                batch_size = batch,
                                                class_mode = class_mode
                                                )
    label_map=(test_set.class_indices)
    print(label_map)

    Y_pred = model.predict(test_set, len(test_set))
    y_pred = np.argmax(Y_pred, axis=1)
    print("Confusion Matrix")
    sns.heatmap(confusion_matrix(test_set.classes,y_pred),annot=True)
    plt.xlabel('Actual vlaues,0:Bulbasaur, 1:Charmander, 2: Squirtle,3:Tauros')
    plt.ylabel('Predicted Value,0:Bulbasaur, 1:Charmander, 2: Squirtle,3:Tauros')
    plt.savefig('reports/Confusion_Matrix')
    #plt.show()

    print("Classification Report")
    target_names = ['Bulbasaur','Charmander','Squirtle','Tauros']
    df =pd.DataFrame(classification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True)).T
    df['support']=df.support.apply(int)
    df.style.background_gradient(cmap='viridis',subset=pd.IndexSlice['0':'9','f1-score'])
    df.to_csv('reports/classification_report')
    print('Classification Report and Confusion Matrix Report are saved in reports folder of Template')



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config',default='params.yaml')
    passed_args = args_parser.parse_args()
    m_evaluate(config_file=passed_args.config)