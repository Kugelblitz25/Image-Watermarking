import numpy as np
import os
import matplotlib.pyplot as plt
from main import Encoder, Decoder, ImageLoader
import cv2

plt.style.use('dark_background')

class Metrics:
    """
    Class for computing image quality metrics.

    Attributes:
    - None
    """

    def getImages(self, img1, img2):
        """
        Load two images and ensure their sizes match.

        Parameters:
        - img1: str or numpy.ndarray
            If str, the file path of the first image. If numpy.ndarray, the first image itself.
        - img2: str or numpy.ndarray
            If str, the file path of the second image. If numpy.ndarray, the second image itself.

        Returns:
        - Tuple[numpy.ndarray, numpy.ndarray]
            A tuple containing the loaded first and second images.
        """
        img1Path, img1 = ImageLoader(img1)
        img2Path, img2 = ImageLoader(img2)
        assert img1.shape == img2.shape, "Image sizes don't match."
        return img1.astype('float64'), img2.astype('float64')

    def MSE(self, img1, img2):
        """
        Compute the Mean Squared Error (MSE) between two images.

        Parameters:
        - img1: str or numpy.ndarray
            If str, the file path of the first image. If numpy.ndarray, the first image itself.
        - img2: str or numpy.ndarray
            If str, the file path of the second image. If numpy.ndarray, the second image itself.

        Returns:
        - float
            The computed MSE value.
        """
        img1, img2 = self.getImages(img1, img2)
        h, w = img1.shape
        return ((img1 - img2) ** 2).sum() / (w * h)

    def PSNR(self, img1, img2):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

        Parameters:
        - img1: str or numpy.ndarray
            If str, the file path of the first image. If numpy.ndarray, the first image itself.
        - img2: str or numpy.ndarray
            If str, the file path of the second image. If numpy.ndarray, the second image itself.

        Returns:
        - float
            The computed PSNR value.
        """
        return 10 * np.log10(255 ** 2 / self.MSE(img1, img2))

    def NCC(self, img1, img2):
        """
        Compute the Normalized Cross-Correlation (NCC) between two images.

        Parameters:
        - img1: str or numpy.ndarray
            If str, the file path of the first image. If numpy.ndarray, the first image itself.
        - img2: str or numpy.ndarray
            If str, the file path of the second image. If numpy.ndarray, the second image itself.

        Returns:
        - float
            The computed NCC value.
        """
        img1, img2 = self.getImages(img1, img2)
        h, w = img1.shape
        CC = ((img1 - img1.mean()) * (img2 - img2.mean())).sum() / (w * h)
        return CC / (img1.std() * img2.std())

class Tester:
    """
    Class for testing the encoder and decoder with various parameters.

    Attributes:
    - imageDir: str
        Directory containing the original images.
    - watermarkDir: str
        Directory containing the watermarks.
    - plane: int
        Bit plane used for encoding/decoding.
    - blocksize: int
        Size of each block for encoding/decoding.
    """

    def __init__(self, imageDir: str, watermarkDir: str, plane: int = 1, blocksize: int = 2) -> None:
        """
        Initialize the Tester with the specified directories and parameters.

        Parameters:
        - imageDir: str
            Directory containing the original images.
        - watermarkDir: str
            Directory containing the watermarks.
        - plane: int, optional
            Bit plane used for encoding/decoding. Default is 1.
        - blocksize: int, optional
            Size of each block for encoding/decoding. Default is 2.
        """
        self.imageDir = imageDir
        self.watermarkDir = watermarkDir
        self.images = os.listdir(self.imageDir)
        self.watermarks = os.listdir(self.watermarkDir)
        self.metrics = Metrics()
        self.encoder = Encoder(blocksize, plane)
        self.decoder = Decoder(blocksize, plane)

    def unitTest(self, imageIdx: int, watermarkIdx: int, disp: bool = False):
        """
        Perform a unit test with a specific pair of images and watermarks.

        Parameters:
        - imageIdx: int
            Index of the image in the 'images' directory.
        - watermarkIdx: int
            Index of the watermark in the 'watermarks' directory.

        Returns:
        - Tuple[float, float]
            A tuple containing the computed PSNR and NCC values.
        """
        _, img = ImageLoader(os.path.join(
            self.imageDir, self.images[imageIdx]))
        _, wm = ImageLoader(os.path.join(self.watermarkDir,
                            self.watermarks[watermarkIdx]))

        encoded_image = self.encoder.encode(img, wm)

        noise = 0.5 * np.random.randn(*encoded_image.shape)
        encoded_image_gaussian_noise = np.uint8(np.round(encoded_image + noise, 0))

        decoded_wm = self.decoder.decode(encoded_image_gaussian_noise)

        if disp:
            cv2.imshow('Decoded',decoded_wm)
            cv2.imshow('Encoded',encoded_image)
            cv2.waitKey(0)

        psnr = self.metrics.PSNR(img, encoded_image)
        ncc = self.metrics.NCC(self.encoder.watermark, decoded_wm)

        return psnr, ncc

    def CompleteTest(self):
        """
        Perform a complete test for all combinations of images and watermarks.

        Returns:
        - Tuple[float, float]
            A tuple containing the average PSNR and NCC values across all pairs.
        """
        PSNR_Tot = 0
        NCC_Tot = 0
        num_pairs = 0

        for i in range(len(self.images)):
            for j in range(len(self.watermarks)):
                psnr, ncc = self.unitTest(i, j)
                PSNR_Tot += psnr
                NCC_Tot += ncc
                num_pairs += 1

        return PSNR_Tot / num_pairs, NCC_Tot / num_pairs

def plotRes(ax, x, y, title, xlabel, ylabel):
    ax.plot(x, y, marker='o')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel,  fontsize=12)
    ax.set_xticks(x)
    ax.grid(axis='x',linestyle='--',color='grey')

def testPlanes(fig,axes):
    # Testing for encoding in different planes with blocksize=4 
    planes = np.arange(1,9)
    PSNRs = []
    NCCs = []

    for plane in planes:
        tester = Tester(imgDir, wmDir, plane, 4)
        psnr, ncc = tester.CompleteTest()
        PSNRs.append(psnr)
        NCCs.append(ncc)

    ax=axes.flatten()
    fig.suptitle('Encoding at different planes.', fontsize=16, fontweight='bold')
    plotRes(ax[0],planes,PSNRs,'Effect on PSNR.','Plane used for encoding','PSNR between normal and encoded image.')
    plotRes(ax[1],planes,NCCs,'Effect on NCC.','Plane used for encoding','NCC between normal and extracted watermark.')
    ax[1].set_ylim(0.5,1)

def testBlocksizes(fig,axes):
    # Testing for encoding with different blocksizes at plane=2 
    blocksizes = np.arange(1,11)
    PSNRs = []
    NCCs = []

    for bs in blocksizes:
        tester = Tester(imgDir, wmDir, 2, bs)
        psnr, ncc = tester.CompleteTest()
        PSNRs.append(psnr)
        NCCs.append(ncc)

    ax=axes.flatten()
    fig.suptitle('Encoding with different blocksizes.', fontsize=16, fontweight='bold')
    plotRes(ax[0],blocksizes,PSNRs,'Effect on PSNR.','Blocksize used for encoding','PSNR between normal and encoded image.')
    plotRes(ax[1],blocksizes,NCCs,'Effect on NCC.','Blocksize used for encoding','NCC between normal and extracted watermark.')
    ax[1].set_ylim(0.5,1)

if __name__=="__main__":
    # Testing
    imgDir = 'imgs'
    wmDir = 'watermarks'

    fig1,axes1=plt.subplots(1,2, figsize=(15,7), sharex=True)
    testPlanes(fig1,axes1)

    fig2,axes2=plt.subplots(1,2, figsize=(15,7), sharex=True)
    testBlocksizes(fig2,axes2)

    # Saving Results
    fig1.savefig('results/planes.png')
    fig2.savefig('results/blocks.png')

    plt.show()
