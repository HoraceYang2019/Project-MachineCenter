import os
from config import initialize_config
from data_processing import main as data_processing
from feature_extraction import main as feature_extraction
from theoretical_estimation import main as theoretical_estimation
from train_pgnn import main as train
from predict_pgnn import main as predict
from fine_tune_pgnn import main as fine_tune

os.chdir(os.path.dirname(__file__))

data = 'data_C'
initialize_config(data)
# data_processing(data)
# feature_extraction(data)
# theoretical_estimation(data)

# train(data)
predict(data)

data = 'data_QIT'
# initialize_config(data)
# theoretical_estimation(data)
predict(data)
fine_tune(data, fine_tune_samples=10, fine_tune_epochs=1000, fine_tune_lr=0.000005)