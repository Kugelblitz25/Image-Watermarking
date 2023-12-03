import numpy as np
import cv2
import matplotlib.pyplot as plt
from main import Encoder, Decoder

def MSE(img1,img2):
    assert img1.shape==img2.shape, "Image sizes doesn't match."
    w,h=img1.shape
    return ((img1-img2)**2).sum()/(w*h)

def PSNR(img1,img2):
    return 10*np.log10(255*255/MSE(img1,img2))

def NCC(img1,img2):
    assert img1.shape==img2.shape, "Image sizes doesn't match."
    w,h=img1.shape
    CC=((img1-img1.mean())*(img2-img2.mean())).sum()/(w*h)
    return CC/(img1.std()*img2.std())

def test(imgPath,wmPath, enc_plane=0, block_size=2):
    img=cv2.imread(imgPath,0)
    wm=cv2.imread(wmPath,0)

    encoder=Encoder(img,wm,block_size,enc_plane) 
    encoded_image=encoder.encode()
    wm=encoder.watermark

    noise=0.5*np.random.randn(*encoded_image.shape)
    encoded_image_gaussian_noise=np.uint8(np.round(encoded_image+noise,0))

    decoder=Decoder(encoded_image_gaussian_noise,block_size,enc_plane)
    decoded_watermark=decoder.decode()

    # cv2.imshow('Decoded',decoded_watermark)
    # cv2.waitKey(0)

    return PSNR(img,encoded_image),NCC(wm,decoded_watermark)

    # print('PSNR between Original and Encoded Image:',PSNR(img,encoded_image))
    # print('NCC between Original and Decoded Watermark:',NCC(wm,decoded_watermark))


img='imgs/barbara256.png'
wm1='watermarks/10.png'
wm2='watermarks/watermark.jpg'

PSNRs=[]
NCCs=[]

for i in range(1,6):
    psnr,ncc=test(img,wm1,block_size=i)
    PSNRs.append(psnr)
    NCCs.append(ncc)

x=np.arange(1,6)
plt.plot(x,NCCs)
plt.show()