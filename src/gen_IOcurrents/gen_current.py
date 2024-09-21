import numpy as np
import matplotlib.pyplot as plt

def generate_correlated_arrays(rows, cols, correlation):
    # Generate an initial random array of the given shape
    arr = np.random.uniform(-1, 1, (rows, cols))

    # Adjust rows to have zero mean
    arr -= arr.mean(axis=1, keepdims=True)

    # Create the correlation matrix
    corr_matrix = correlation * np.ones((rows, rows)) + (1 - correlation) * np.eye(rows)

    # Perform Cholesky decomposition to obtain the transformation matrix
    L = np.linalg.cholesky(corr_matrix)

    # Apply the transformation to introduce the desired correlation
    correlated_arr = np.dot(L, arr)

    # Rescale each row to fit the range [-1, 1] while keeping the mean zero
    min_vals = correlated_arr.min(axis=1, keepdims=True)
    max_vals = correlated_arr.max(axis=1, keepdims=True)
    scale = np.maximum(max_vals, -min_vals) / 1
    correlated_arr = correlated_arr / scale

    return correlated_arr

# Parameters
rows = 13
cols = 9
correlation = 0.3 # modify this to get different correlation currents
I_mean = 1.3
ripple = 0.07

# Generate the array
result_array = I_mean*ripple*generate_correlated_arrays(rows, cols, correlation)


# Generate the plot
plt.figure(figsize=(10, 6))
for i in range(result_array.shape[0]):
    plt.plot(result_array[i], label=f'Row {i+1}')

loc = [(666, 8666), (2000, 8666), (3333, 8666), (4666, 8666), (6000, 8666), (7333, 8666), (8666, 8666), (8666, 666), (8666, 2000), (8666, 3333), (8666, 4666), (8666, 6000), (8666, 7333)]
strs = ''
for i in range(rows):
    s = 'id_add_%d nd_1_0_%d_%d 0 pwl(0 0 0.1ns %.3f 0.2ns %.3f 0.3ns %.3f 0.4ns %.3f 0.5ns %.3f 0.6ns %.3f 0.7ns %.3f 0.8ns %.3f 0.9ns %.3f 1.0ns 0.0)\n' % (
    i, loc[i][0], loc[i][1], result_array[i][0], result_array[i][1], result_array[i][2], result_array[i][3],
    result_array[i][4], result_array[i][5], result_array[i][6], result_array[i][7], result_array[i][8],
    )
    strs += s

for j in range(rows):
    s = str(result_array[j])
    strs += s
    
with open('current_%.2f.txt' % (correlation),'w') as f1:
    f1.writelines(strs)
    f1.close()
    
    
plt.plot(result_array.sum(axis=0), label=f'Sum')
plt.title('Plot of Each Row with Different Colors')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()


        
        
        
        
        
