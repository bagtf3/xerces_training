import os
import numpy as np
from xerces_training.chunkparser import ChunkParser


chunk_dir = "C:/Users/Bryan/Data/chessbot_data/training_data/lc0/training-run2--20200711-2018"
all_files = [f for f in os.listdir(chunk_dir) if f.endswith(".gz")]
import random; random.shuffle(all_files)
chunks = [os.path.join(chunk_dir, c) for c in all_files[:20]]
batch_size = 16
# Make a parser that does NOT spawn worker processes or the chunk_reader
parser = ChunkParser(
    chunks,
    expected_input_format=1,  # match how the chunks were written (1/2/3/etc)
    shuffle_size=1,
    sample=1,
    buffer_size=1,
    batch_size=batch_size,
    workers=0,                # <= IMPORTANT: disables chunk_reader / workers
)


for batch in parser.sequential():
    
    tokens = batch[0]
    masks = batch[1]
    policies = batch[2]
    ys = batch[3]
    fens = batch[4]

