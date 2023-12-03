import cv2
import numpy as np

class Encoder:
    def __init__(self, img, watermark, blockSize=2, plane=1):
        self.plane=plane
        self.blockSize=blockSize

        self.img=img
        if type(img)==str:
            self.imgPath=img
            self.img=cv2.imread(self.imgPath,0)
        self.h,self.w=self.img.shape

        try:
            self.watermark=watermark
        except:
            self.watermarkPath=watermark
            self.watermark=cv2.imread(self.watermarkPath,0)
        self.watermark=cv2.resize(self.watermark,(self.h//self.blockSize,self.w//self.blockSize))
        _,self.watermark=cv2.threshold(self.watermark,127,255,cv2.THRESH_BINARY)
        self.h_w,self.w_w=self.watermark.shape

        self.img.dtype=np.uint8
        self.watermark.dtype=np.uint8

        assert (self.h_w==self.h//self.blockSize and self.w_w==self.w//self.blockSize),\
                f'Image and watermark shapes of {self.img.shape} and {self.watermark.shape} are in compatible.'
    
    def encodeBlockBit(self,row,col):
        enc=self.watermark[row,col]
        img_block=self.img[row*self.blockSize:(row+1)*self.blockSize,
                           col*self.blockSize:(col+1)*self.blockSize]
        enc_block=enc*np.ones_like(img_block,'uint8')
        masked_img=img_block & (255 - 1<<self.plane)
        masked_enc=enc_block & (1<<self.plane)
        return masked_img+masked_enc
    
    def encode(self):
        encodedImg=np.zeros_like(self.img)
        for col in range(self.w_w):
            for row in range(self.h_w):
                encodedImg[row*self.blockSize:(row+1)*self.blockSize,
                           col*self.blockSize:(col+1)*self.blockSize]=self.encodeBlockBit(row,col)
        return encodedImg.astype('uint8')
    

class Decoder:
    def __init__(self,img, blockSize=2, plane = 1):
        self.img=img
        self.plane=plane
        self.blockSize=blockSize
        if type(img)==str:
            self.imgPath=img
            self.img=cv2.imread(self.imgPath,0)
        self.h,self.w=self.img.shape

    def decode(self):
        self.h_w,self.w_w=self.h//self.blockSize,self.w//self.blockSize
        watermark=np.zeros((self.h_w,self.w_w))
        img_plane=self.img & (1<<self.plane)
        for col in range(self.w_w):
            for row in range(self.h_w):
                    img_block=img_plane[row*self.blockSize:(row+1)*self.blockSize,
                              col*self.blockSize:(col+1)*self.blockSize]
                    # print(img_block)
                    enc_byte=255*np.round(img_block.mean(),0)
                    watermark[row,col]=enc_byte
        return watermark.astype('uint8')


