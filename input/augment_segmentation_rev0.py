import os
import Augmentor
import argparse
from Augmentor.Operations import Operation
from PIL import ImageFilter

parser = argparse.ArgumentParser(description='TGS Training Data Augmentation Generation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output-dir',
                    help='output directory')
parser.add_argument('-N', '--sample-number', type=int,
                    help='number of samples')
parser.add_argument('-W', '--width', type=int, default=101,
                    help='resize_width')
parser.add_argument('-H', '--height', type=int, default=101,
                    help='resize height')

def main():
    global args, best_prec1
    args = parser.parse_args()

    if os.path.isabs(args.output_dir):
        outpath = args.output_dir
    else:
        outpath = os.getcwd() + "/" + args.output_dir

    if os.path.isabs(args.data):
        inppath = args.data
    else:
        inppath = os.getcwd() + "/" + args.data
        
    
    print("Input directory: {}".format(inppath))
    print("Output directory: {}".format(outpath))
    print("Number of samples to generate: {}".format(args.sample_number))

    # create model
    p = Augmentor.Pipeline(inppath + '/images', save_format="png", output_directory=outpath)
    p.ground_truth(inppath + '/masks')
    p.flip_random(probability=0.67)
    p.rotate_random_90(probability=0.75)
    p.zoom_random(probability=0.50, percentage_area=0.90, randomise_percentage_area=True)
    p.rotate(probability=0.50, max_left_rotation=5, max_right_rotation=5)
    #p.rotate_without_crop(probability=1.0, max_left_rotation=22.5, max_right_rotation=22.5)    
    #p.skew(probability=1.0, magnitude=0.20)
    #p.shear(probability=1.0, max_shear_left=7.5, max_shear_right=7.5)
    p.random_distortion(probability=0.50, grid_width=10, grid_height=10, magnitude=10)
    #p.resize(probability=1.0, width=args.width, height=args.height)
    #p.random_contrast(probability=1.0, min_factor=0.75, max_factor=1.25)
    #p.random_brightness(probability=1.0, min_factor=0.75, max_factor=1.25)
    #p.random_color(probability=1.0, min_factor=0.75, max_factor=1.25)

    #p.add_operation(BlurImage(probability=1.0, radius=4))

    print(p.status())
    
    p.sample(args.sample_number)        

if __name__ == '__main__':
    main()

