# Rec_code_data

## Code
The code implement the models like LightGCN, MF, LightGCL, XSimGCL, SimGCL, and implement various loss function like BPR Loss, BCE Loss, Softmax Loss. Pay attention to how to implement in an efficient way, such as negative sampling.

## Data
Each dataset contains two txt files, representing the test set and the training set respectively. The number 'k' after '_' in filename represents k-core processed. In txt file, the first line represents the user ID, and the second line represents the item ID. All IDs are serialized. 
