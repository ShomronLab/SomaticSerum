# Somatic Variant Detection in Cell-Free Tumor DNA Using Deep Learning Sequence Read Classification Approach
This repository holds code for processing and applying several deep learning model on cell-free DNA of cancer patients.
This work is part of DLC21-22 taken at CS, TAU

# Install
All needed python dependcies relevnet for the bioinformatic and deep learning can be obtained via conda and specified in SomaticSerum_env.yml
``` bash
conda create -f SomaticSerum_env.yml
```
# Train/test a model
``` python
python main.py [--sample_split SAMPLE_SPLIT] [--model MODEL] [--hidden_size HIDDEN_SIZE] [--sequence_length SEQUENCE_LENGTH] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--max_epoch MAX_EPOCH] [--lstm_layers LSTM_LAYERS] [--dropout DROPOUT] [--num_workers NUM_WORKERS] [--out OUT] [--test TEST]
                training_bam_dir

positional arguments:
  training_bam_dir      Train data bams directory

optional arguments:
  -h, --help            show this help message and exit
  --sample_split SAMPLE_SPLIT
                        How to split the training data: True - by samples, False - by random on the entire dataset
  --model MODEL         model
  --hidden_size HIDDEN_SIZE
                        The number of hidden units
  --sequence_length SEQUENCE_LENGTH
                        The length of the sequence
  --batch_size BATCH_SIZE
                        The size of each batch
  --learning_rate LEARNING_RATE
                        The learning rate value
  --max_epoch MAX_EPOCH
                        The maximum epoch
  --lstm_layers LSTM_LAYERS
                        Num of LSTM layers
  --dropout DROPOUT     Dropout
  --num_workers NUM_WORKERS
                        Number of workers
  --out OUT             Output directory
  --test TEST           Test directory
```