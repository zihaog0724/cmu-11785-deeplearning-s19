import os
import shutil

cur_dir = os.getcwd() + '/train_data/medium'
list_dir = os.listdir(cur_dir)
n = 0
for sub_dir in list_dir:
	dir_to_move = os.getcwd() + '/train_data/large/'
	if sub_dir.endswith('Store') is not True:
		shutil.move(cur_dir + '/' + sub_dir, dir_to_move)
	n += 1
	print(n)
