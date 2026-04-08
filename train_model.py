import argparse
from neural_decoder.neural_decoder_trainer import trainModel

# --- Hyperparameters ---
args = {}
args['batchSize'] = 16
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 512        
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4       
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200

# --- Paths (pass via CLI or edit directly) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GRU phoneme decoder for Speech BCI')
    parser.add_argument('--output_dir',   type=str, required=True, help='Directory to save model weights and training stats')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to preprocessed ptDecoder_ctc pickle')
    cli = parser.parse_args()

    args['outputDir']   = cli.output_dir
    args['datasetPath'] = cli.dataset_path

    trainModel(args)
