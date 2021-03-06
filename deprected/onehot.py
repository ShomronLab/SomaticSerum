from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re

class hot_dna:
    def __init__(self,fasta):
   
        #check for and grab sequence name
        if re.search(">",fasta):
            name = re.split("\n",fasta)[0]
            sequence = re.split("\n",fasta)[1]
        else :
            name = 'unknown_sequence'
            sequence = fasta
        
        #get sequence into an array
        seq_array = array(list(sequence))
            
        #integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)
            
        #one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        #reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        
        #add the attributes to self 
        self.name = name
        self.sequence = fasta
        self.integer = integer_encoded_seq
        self.onehot = onehot_encoded_seq


if "__main__" == __name__:
    fasta = ">fake_sequence\nTGTGTCGTGTCGTCG"
    my_hottie = hot_dna(fasta)

    print(my_hottie.name)
    print(my_hottie.sequence)
    print(my_hottie.integer)
    print(my_hottie.onehot)