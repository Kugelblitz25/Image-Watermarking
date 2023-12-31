"""
This file contains the testing infrastructure for Image Watermarking algorithm.

Author: Vighnesh Nayak
Date: 7 Dec 2023
Github: https://github.com/Kugelblitz25
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from main import Decoder, Encoder, Image

plt.style.use("dark_background")


def getArgs():
    """
    Parse command-line arguments for the script.

    Returns:
    - argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Directory of test images")
    parser.add_argument("-w", "--watermark_dir", type=str, required=True, help="Directory of test watermarks")
    parser.add_argument(
        "-o",
        "--result_path",
        type=str,
        required=False,
        default="results",
        help="Directory where resulting graph needs to be written.",
    )
    parser.add_argument(
        "-n",
        "--noise_type",
        choices=["Gaussian", "SnP"],
        required=False,
        default="Gaussian",
        help="Type of noise to be tested under.",
    )
    parser.add_argument(
        "-s", "--strength", nargs="*", type=float, default=[0, 0.5], required=False, help="Strength of noise used."
    )
    parser.add_argument(
        "-t",
        "--type_of_test",
        choices=["single", "unit", "complete"],
        required=False,
        default="unit_test",
        help="Type of test to be conducted (over all images or one single image).",
    )
    parser.add_argument(
        "-p",
        "--param",
        choices=["planes", "blocksize", "both"],
        required=False,
        default="planes",
        help="Which parameters to be tested.",
    )

    return parser.parse_args()


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
        img1 = Image(img1)
        img2 = Image(img2)
        assert all(img1.imageSize == img2.imageSize), "Image sizes don't match."
        return img1.image.astype("float64"), img2.image.astype("float64")

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
        return 10 * np.log10(255**2 / self.MSE(img1, img2))

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
    def __init__(
        self, imageDir: str, watermarkDir: str, plane: int, blocksize: int, noise: str, strength: list[int]
    ) -> None:
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
        - noise: str, optional
            Type of noise to be applied (Gaussian or SnP). Default is Gaussian.
        - strength: list[int], optional
            Strength of the applied noise. Default is [0, 0.5].
        """
        self.imageDir = imageDir
        self.watermarkDir = watermarkDir
        self.images = os.listdir(self.imageDir)
        self.watermarks = os.listdir(self.watermarkDir)
        self.noiseType = noise
        self.noiseStrength = strength
        self.metrics = Metrics()
        self.encoder = Encoder(blocksize, plane)
        self.decoder = Decoder(blocksize, plane)

    def getNoise(self, shape: tuple[int]) -> np.ndarray:
        """
        Generate and return the specified type of noise.

        Parameters:
        - shape: Tuple
            Shape of the noise array.

        Returns:
        - numpy.ndarray
            The generated noise array.
        """
        if self.noiseType == "Gaussian":
            noise = np.random.normal(*self.noiseStrength[:2], shape)
            return np.uint8(np.round(noise, 0))

        if self.noiseType == "SnP":
            noise = np.random.choice([0, 255], shape, True, [1 - self.noiseStrength[0], self.noiseStrength[0]])
            return np.uint8(noise)

    def unitTest(self, imageIdx: int, watermarkIdx: int, disp: bool = False) -> tuple[float, float]:
        """
        Perform a unit test with a specific pair of images and watermarks.

        Parameters:
        - imageIdx: int
            Index of the image in the 'images' directory.
        - watermarkIdx: int
            Index of the watermark in the 'watermarks' directory.
        - disp: bool, optional
            Whether to display the images during testing. Default is False.

        Returns:
        - Tuple[float, float]
            A tuple containing the computed PSNR and NCC values.
        """
        img = Image(os.path.join(self.imageDir, self.images[imageIdx])).image
        wm = Image(os.path.join(self.watermarkDir, self.watermarks[watermarkIdx])).image

        encodedImage = self.encoder.encode(img, wm)

        encodedImageGaussianNoise = encodedImage + self.getNoise(encodedImage.shape)

        decodedWM = self.decoder.decode(encodedImageGaussianNoise)

        if disp:
            cv2.imshow("Decoded", decodedWM)
            cv2.imshow("Encoded", encodedImage)
            cv2.waitKey(0)

        psnr = self.metrics.PSNR(img, encodedImage)
        ncc = self.metrics.NCC(self.encoder.watermark, decodedWM)

        return psnr, ncc

    def completeTest(self) -> tuple[float, float]:
        """
        Perform a complete test for all combinations of images and watermarks.

        Returns:
        - Tuple[float, float]
            A tuple containing the average PSNR and NCC values across all pairs.
        """
        PSNR_Tot = 0
        NCC_Tot = 0
        numPairs = 0

        for i in range(len(self.images)):
            for j in range(len(self.watermarks)):
                psnr, ncc = self.unitTest(i, j)
                PSNR_Tot += psnr
                NCC_Tot += ncc
                numPairs += 1

        return PSNR_Tot / numPairs, NCC_Tot / numPairs


def plotRes(ax, x, y, title, xlabel, ylabel):
    """
    Plot the resluts.
    """
    ax.plot(x, y, marker="o")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.grid(axis="x", linestyle="--", color="grey")


def testPlanes(imgDir, wmDir, fig, axes, typeOfTest, noise, strength):
    """
    Testing for effect of encoding in different planes with blocksize=2
    """
    planes = np.arange(1, 9)
    PSNRs = []
    NCCs = []

    for plane in planes:
        tester = Tester(imgDir, wmDir, plane, 2, noise, strength)
        if typeOfTest == "complete":
            psnr, ncc = tester.completeTest()
        if typeOfTest == "unit":
            psnr, ncc = tester.unitTest(0, 0)
        PSNRs.append(psnr)
        NCCs.append(ncc)

    ax = axes.flatten()
    fig.suptitle("Encoding at different planes.", fontsize=16, fontweight="bold")
    plotRes(
        ax[0], planes, PSNRs, "Effect on PSNR.", "Plane used for encoding", "PSNR between normal and encoded image."
    )
    plotRes(
        ax[1], planes, NCCs, "Effect on NCC.", "Plane used for encoding", "NCC between normal and extracted watermark."
    )
    ax[1].set_ylim(0.5, 1)


def testBlocksizes(imgDir, wmDir, fig, axes, typeOfTest, noise, strength):
    """
    Testing for effects of encoding with different blocksizes at plane=2
    """
    blocksizes = np.arange(1, 11)
    PSNRs = []
    NCCs = []

    for bs in blocksizes:
        tester = Tester(imgDir, wmDir, 2, bs, noise, strength)
        if typeOfTest == "complete":
            psnr, ncc = tester.completeTest()
        if typeOfTest == "unit":
            psnr, ncc = tester.unitTest(0, 0)
        PSNRs.append(psnr)
        NCCs.append(ncc)

    ax = axes.flatten()
    fig.suptitle("Encoding with different blocksizes.", fontsize=16, fontweight="bold")
    plotRes(
        ax[0],
        blocksizes,
        PSNRs,
        "Effect on PSNR.",
        "Blocksize used for encoding",
        "PSNR between normal and encoded image.",
    )
    plotRes(
        ax[1],
        blocksizes,
        NCCs,
        "Effect on NCC.",
        "Blocksize used for encoding",
        "NCC between normal and extracted watermark.",
    )
    ax[1].set_ylim(0.5, 1)


if __name__ == "__main__":
    # Testing
    opts = getArgs()
    imgDir = opts.image_dir
    wmDir = opts.watermark_dir

    if opts.param == "planes" or opts.param == "both":
        if opts.type_of_test == "single":
            tester = Tester(imgDir, wmDir, 2, 3, opts.noise_type, opts.strength)
            psnr, ncc = tester.unitTest(0, 0, True)
            print("PSNR between original and encoded image:", psnr)
            print("NCC between original and extracted watermark:", ncc)
        else:
            fig1, axes1 = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
            testPlanes(imgDir, wmDir, fig1, axes1, opts.type_of_test, opts.noise_type, opts.strength)
            fig1.savefig(opts.result_path + "/planes.png")

    if opts.param == "blocksize" or opts.param == "both":
        if opts.type_of_test == "single":
            tester = Tester(imgDir, wmDir, 2, 3, opts.noise_type, opts.strength)
            psnr, ncc = tester.unitTest(0, 0, True)
            print("PSNR between original and encoded image:", psnr)
            print("NCC between original and extracted watermark:", ncc)
        else:
            fig2, axes2 = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
            testBlocksizes(imgDir, wmDir, fig2, axes2, opts.type_of_test, opts.noise_type, opts.strength)
            fig2.savefig(opts.result_path + "/blocks.png")

    plt.show()
