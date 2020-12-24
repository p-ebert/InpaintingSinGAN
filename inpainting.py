from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import colour_fill
import SinGAN.functions as functions
import torch
import cv2

if __name__=="__main__":
    parser=get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--on_drive', help='using drive or not', default=None)
    parser.add_argument('--mask_name', help='name of mask image', required=True)
    parser.add_argument('--inpainting_scale_start', default=1)
    parser.add_argument('--radius', default=7)
    opt = parser.parse_args()
    opt.mode = "inpainting_generate"
    opt.inpainting_scale_start=int(opt.inpainting_scale_start)
    opt = functions.post_config(opt)

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_scale_start < 1) | (opt.inpainting_scale_start > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            #Importing masked image
            if opt.on_drive!=None:
                masked_img = cv2.imread('%s/%s/%s' % (opt.on_drive, opt.input_dir, opt.input_name))
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            else:
                masked_img = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))
                masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

            #Importing mask
            if opt.on_drive!=None:
                mask = cv2.imread('%s/%s/%s' % (opt.on_drive, opt.input_dir, opt.mask_name))
            else:
                mask = cv2.imread('%s/%s' % (opt.input_dir, opt.mask_name))

            #Converting to binary mask
            mask=1-mask/255

            coloured_image=colour_fill.mean_colour_rectangular(masked_img, mask)

            #writing coloured img
            if opt.on_drive!=None:
                cv2.imwrite("%s/%s/%s_coloured.jpg" % (opt.on_drive, dir2save, opt.mask_name[:-4]),coloured_image)
            else:
                cv2.imwrite("%s/%s_coloured.jpg" % (dir2save, opt.mask_name[:-4]), coloured_image)

            #Reading in coloured image
            if opt.on_drive!=None:
                ref = functions.read_image_dir('%s/%s/%s_coloured.jpg' % (opt.on_drive, dir2save, opt.mask[:-4]), opt)
                mask = functions.read_image_dir('%s/%s/%s' % (opt.on_drive, opt.input_dir,opt.mask_name), opt)

            else:
                ref = functions.read_image_dir('%s/%s_coloured.jpg' % (dir2save, opt.mask_name[:-4]), opt)
                mask = functions.read_image_dir('%s/%s' % (opt.input_dir,opt.mask_name), opt)

            #ref=cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))
            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]

            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_scale_start
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out

            out = functions.convert_image_np(out.detach())

            plt.imshow(out)
            plt.show()

            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)*255

            cv2.imwrite('%s/%s_start_scale=%d.jpg' % (dir2save,opt.input_name[:-4],opt.inpainting_scale_start), out)
