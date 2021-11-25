from tensorboard.plugins.hparams import api as hp

# Preprocessing params
DataPath = "/data"
DevDataSetPath = "/data/aidatatang_200zh/corpus/dev"
DataSetName = "aidatatang_200zh"
DevDataPath = "corpus/dev"
DevDataDir = "corpus/dev/G0002"
PrefixName = "T0055G0002S0002"

# Preprocessing Hparams
HP_SAMPLE_RATE = hp.HParam('sample_rate', hp.Discrete([16000]))
HP_PRE_EMPHASIS = hp.HParam('pre_emphasis', hp.Discrete([0.97]))
HP_MEL_BINS = hp.HParam('mel_bins', hp.Discrete([32]))
HP_FRAME_LENGTH = hp.HParam('frame_length', hp.Discrete([0.025]))
HP_FRAME_STEP = hp.HParam('frame_step', hp.Discrete([0.01]))
HP_HERTZ_LOW = hp.HParam('hertz_low', hp.Discrete([125.0]))
HP_HERTZ_HIGH = hp.HParam('hertz_high', hp.Discrete([7600.0]))
HP_DOWNSAMPLE_FACTOR = hp.HParam('downsample_factor', hp.Discrete([3]))

# Model Hparams
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1]))
HP_MEL_STEP = hp.HParam('mel_step', hp.Discrete([8]))
HP_CNN_LAYERS = hp.HParam('cnn_layers', hp.Discrete([8]))
HP_CNN_FILTER = hp.HParam('cnn_filter', hp.Discrete([9]))