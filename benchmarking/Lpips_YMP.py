import argparse
import os
from lpips import lpips
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='../datasets/100-shot-obama/img')
parser.add_argument('-d1','--dir1', type=str, default='../datasets/100-shot-panda/img')
parser.add_argument('-o','--out', type=str, default='./example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', default='TRUE', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

output_name = '../train_results/panda_test/lpips.txt'

# if not os.path.exists(output_name):
# 	os.mknod('./lpips.txt')

# crawl directories
f = open(output_name,'w')
files = os.listdir(opt.dir0)
sum = 0.0
count = 0
for file in files:
	if(os.path.exists(os.path.join(opt.dir1,file))):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		print('%s: %.3f'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,dist01))
		sum = sum + dist01
		count+=1
f.writelines('%s:%.6f\n'%("The Sum of all dis:",sum))
f.writelines('%s:%.6f\n'%("The number of computed images:",count))
f.writelines('%s:%.6f\n'%("The final Avg results:",sum/count))
print(sum/count)

f.close()
