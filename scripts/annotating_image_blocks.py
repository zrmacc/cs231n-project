import os
import tifffile as tf
import numpy as np

image_dir = './'
file = 'PR159_h3_working.tif'

arr = tf.imread(os.path.join(image_dir,file))
data = np.array(arr)
data = data.transpose()
# np.save(image_dir+'/'+'NPY/'+file.split('.')[0]+'npy',data)

#load csv of collateral coordinates and store in np array in the format (C,P,XYZ)
#where C is the collaterl bridge, P is the point (either 0 or 1), and XYZ are the coordinates
my_data = np.genfromtxt('Collateral_bridge_coords.csv', delimiter=',')
collateral_coordinates = my_data.reshape(int(my_data.shape[0]/2),2,3)

W,H,D = data.shape
BLOCK_SIZE = np.array([256,256,128])
STRIDE = np.array([128,128,64])
THRESHOLD = 50

padded_data = np.pad(data,((STRIDE[0],STRIDE[0]),(STRIDE[1],STRIDE[1]),(STRIDE[2],STRIDE[2])),mode='constant')

num_blocks = (data.shape-BLOCK_SIZE)/STRIDE + 1
num_blocks = np.ceil(num_blocks).astype(int)
print(num_blocks)

image_blocks = np.zeros((num_blocks[0],num_blocks[1],num_blocks[2],BLOCK_SIZE[0],BLOCK_SIZE[1],BLOCK_SIZE[2]))
block_col_count = np.zeros(num_blocks)

#splitting image into blocks and counting number of collateral bridges in each block
for i in range(image_blocks.shape[0]):
	for j in range(image_blocks.shape[1]):
		for k in range(image_blocks.shape[2]):
			x_idx = (i+1)*STRIDE[0]
			y_idx = (j+1)*STRIDE[1]
			z_idx = (k+1)*STRIDE[2]
			image_mean = np.mean(padded_data[x_idx:x_idx+BLOCK_SIZE[0],y_idx:y_idx+BLOCK_SIZE[1],z_idx:z_idx+BLOCK_SIZE[2]])
			image_blocks[i,j,k,:,:,:] = padded_data[x_idx:x_idx+BLOCK_SIZE[0],y_idx:y_idx+BLOCK_SIZE[1],z_idx:z_idx+BLOCK_SIZE[2]]
			for c in range(collateral_coordinates.shape[0]):
				max_coords = np.max(collateral_coordinates[c], axis = 0)
				min_coords = np.min(collateral_coordinates[c], axis = 0)
				left = max_coords[0] < x_idx+BLOCK_SIZE[0]
				right = min_coords[0] > x_idx
				above = min_coords[1] > y_idx
				below = max_coords[1] < y_idx+BLOCK_SIZE[1]
				front = min_coords[2] > z_idx
				behind = max_coords[2] < z_idx+BLOCK_SIZE[2]
				if left and right and above and below and front and behind:
					block_col_count[i,j,k] += 1
#block_col_count = np.load('block_col_count.npy')
print(image_blocks[4,4,4,100,100,100])
#determing 45 col negative blocks and 15 col positive blocks (based on block_col_count)
pad = 2
col_neg = []

while len(col_neg)<45:
	i_block = pad+np.random.choice(np.arange(block_col_count.shape[0])-pad)
	j_block = pad+np.random.choice(np.arange(block_col_count.shape[1])-pad)
	k_block = pad+np.random.choice(np.arange(block_col_count.shape[2])-pad)
	if block_col_count[i_block,j_block,k_block]==0:
		col_neg.append([i_block,j_block,k_block])

col_neg_image_blocks = np.zeros((len(col_neg),BLOCK_SIZE[0],BLOCK_SIZE[1],BLOCK_SIZE[2]))
for i in range(len(col_neg)):
	x = col_neg[i][0]
	y = col_neg[i][1]
	z = col_neg[i][2]
	col_neg_image_blocks[i] = image_blocks[x,y,z,:,:,:]
np.save('col_neg_blocks.npy',col_neg_image_blocks)


col_pos = []

while len(col_pos)<15:
	i_block = pad+np.random.choice(np.arange(block_col_count.shape[0])-pad)
	j_block = pad+np.random.choice(np.arange(block_col_count.shape[1])-pad)
	k_block = pad+np.random.choice(np.arange(block_col_count.shape[2])-pad)
	if block_col_count[i_block,j_block,k_block]>0:
		col_pos.append([i_block,j_block,k_block])

col_pos_image_blocks = np.zeros((len(col_pos),BLOCK_SIZE[0],BLOCK_SIZE[1],BLOCK_SIZE[2]))
for i in range(len(col_pos)):
	x = col_pos[i][0]
	y = col_pos[i][1]
	z = col_pos[i][2]
	col_pos_image_blocks[i] = image_blocks[x,y,z,:,:,:]
np.save('col_pos_blocks.npy',col_pos_image_blocks)
#np.save('image_blocks.npy',image_blocks)
#np.save('block_col_count.npy',block_col_count)



