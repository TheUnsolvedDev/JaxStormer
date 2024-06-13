import numpy as np

def replace_with_running_sums(sequence):
    running_sum = 0
    sequence = sequence[::-1]
    for i, val in enumerate(sequence):
        if i < len(sequence)-1:
            if val == 1 and sequence[i+1] == 1 :
                running_sum += 0.9*val
            elif val == 1:
                sequence[i] = val    
            else:
                sequence[i] = running_sum
                running_sum = 0
    return sequence

# Example usage
sequence = [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,0]
new_sequence = replace_with_running_sums(sequence)
print("Original sequence:", sequence)
print("New sequence:", new_sequence)
