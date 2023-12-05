from main import Encoder, Decoder
from tester import Metrics
import re
import argparse
import cv2

def getArgs():
    """
    Parse command-line arguments for the script.

    Returns:
    - argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True,
                        help="Path for the image to be watermarked.")
    parser.add_argument('-w', '--watermark_path', type=str,
                        required=True, help="Path for the watermark.")
    parser.add_argument('-o', '--result_path', type=str, required=False,
                        default='results', help="Directory where watermarked image needs to be stored.")
    parser.add_argument('-p', '--plane', choices=[i+1 for i in range(8)], type=int,
                        required=False, default=1, help='Plane at which encoding is to be done.')
    parser.add_argument('-b', '--block_size', choices=[i+1 for i in range(10)], type=int,
                        required=False, default=3, help='Block size with which encoding is to be done.')

    return parser.parse_args()

if __name__ == "__main__":
    opts = getArgs()
    metric = Metrics()
    img = opts.image_path
    wm = opts.watermark_path
    out_dir = opts.result_path
    blocksize = int(opts.block_size)
    plane = int(opts.plane)

    outImgName = re.split(r"[/\.]+", img)[-2] + '_' + re.split(r"[/\.]+", wm)[-2] + '.png'
    out = out_dir + '/' + outImgName

    encoder = Encoder(blocksize, plane)
    decoder = Decoder(blocksize, plane)

    encoded_img = encoder.encode(img, wm)
    print('PSNR between encoded and original image is:', metric.PSNR(encoder.img, encoded_img))

    cv2.imwrite(out, encoded_img)
    cv2.imshow('Encoded', encoded_img)
    cv2.waitKey(0)
