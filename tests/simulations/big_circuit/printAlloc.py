import gzip
import pickle 
import sys
import numpy as np
import json

def convert_to_standard_types(obj):
        """Converts an object containing defaultdicts of Vectors to standard Python types."""
        result = {}
        for population, vectors in obj.items():
            result[population] = {
                key: np.array(vector) for key, vector in vectors.items()
            }
        return result


filename = sys.argv[1]

with gzip.open(filename, "rb") as f:
    data = pickle.load(f)  # noqa: S301 
    data = convert_to_standard_types(data)
   
    for population, alloc in data.items():
         print(population)
         for rank_cycle, ids in alloc.items():
              print(f"  {rank_cycle}: {ids}")

    
