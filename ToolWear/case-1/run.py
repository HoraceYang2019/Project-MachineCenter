import os
from config import initialize_config
from data_processing import main as data_processing
from feature_extraction import main as feature_extraction
from theoretical_estimation import main as theoretical_estimation
from train_pgnn import main as train
from predict_pgnn import main as predict
from fine_tune_pgnn import main as fine_tune

os.chdir(os.path.dirname(__file__))

data = 'data'
initialize_config(data)
# data_processing(data)
# feature_extraction(data)

# theoretical_estimation(data)
# theoretical_estimation(data, [18])

# train(data)
predict(data)

data = 'data_B'
initialize_config(data)
# data_processing(data)
# feature_extraction(data)

# data_processing(data)
# feature_extraction(data)

# theoretical_estimation(data)
# theoretical_estimation(data, [17])

# train(data)
predict(data)
fine_tune(data, fine_tune_samples=3, fine_tune_epochs=1000, fine_tune_lr=0.0000005)
