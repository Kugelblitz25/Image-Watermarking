"""
This module contains the implimentation of block based Image Watermarking
using bitplane splicing.

Author: Vighnesh Nayak
Date: 2 Dec 2023
Github: https://github.com/Kugelblitz25
"""

import numpy as np
import cv2


class Image:
    def __init__(self, img):
        """
        Load an image from either a file path or a NumPy array.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the image. If numpy.ndarray, the image itself.

        Returns:
        - Tuple[str, numpy.ndarray]
            A tuple containing the image path (if applicable) and the loaded image as a NumPy array.
        """
        if not isinstance(img, (str, np.ndarray)):
            raise ValueError(
                "Invalid input type. `img` must be a string or a numpy array.")

        image = img
        imgPath = None

        if isinstance(img, str):
            imgPath = img
            try:
                image = cv2.imread(imgPath, 0)
            except Exception as e:
                raise ValueError(f"Error loading image: {str(e)}")

        self.image = image.astype(np.uint8)
        self.imageSize = np.array(self.image.shape)

    def reshape(self, shape: list[int]):
        """
        Reshape the image into required size.

        Parameters:
        - shape : List[int]
            Required size of the image.
        """
        self.image = cv2.resize(self.image, shape[::-1])
        self.imageSize = np.array(self.image.shape)

    def binarize(self):
        """
        Binaruze the image into 0's & 1's.
        """
        _, self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        self.image = self.image / 255
        self.image = self.image.astype('uint8')


class Encoder:
    """
    For encoding a watermark into an image using a block-based approach.

    Attributes:
    - blockSize: int
        Size of each block for encoding.
    - plane: int
        Bit plane used for encoding.
    """

    def __init__(self, blockSize: int = 2, plane: int = 1):
        """
        Initialize the Encoder with the specified block size and bit plane.

        Parameters:
        - blockSize: int, optional
            Size of each block for encoding. Default is 2.
        - plane: int, optional
            Bit plane used for encoding. Default is 1.
        """
        self.plane = plane - 1
        self.blockSize = blockSize

    def encodeBlockBit(self, row: int, col: int):
        """
        Encode a single bit of the watermark into a block of the image.

        Parameters:
        - row: int
            Row index of the block.
        - col: int
            Column index of the block.

        Returns:
        - numpy.ndarray
            Encoded block of the image.
        """
        enc = self.watermark[row, col]
        imgBlock = self.img[
            row * self.blockSize: (row + 1) * self.blockSize,
            col * self.blockSize: (col + 1) * self.blockSize
        ]
        encBlock = enc * np.ones_like(imgBlock, "uint8")
        maskedImg = imgBlock & (255 - (1 << self.plane))
        maskedEnc = encBlock & (1 << self.plane)
        return maskedImg + maskedEnc

    def encode(self, img, watermark):
        """
        Encode the watermark into the image.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the image.
            If numpy.ndarray, the image itself.
        - watermark: str or numpy.ndarray
            If str, the file path of the watermark image.
            If numpy.ndarray, the watermark itself.

        Returns:
        - numpy.ndarray
            Encoded image.
        """
        image = Image(img)
        watermark = Image(watermark)
        watermark.reshape(image.imageSize // self.blockSize)
        watermark.binarize()
        self.img, self.imgSize = image.image, image.imageSize
        self.watermark, self.watermarkSize = watermark.image, watermark.imageSize

        assert all(self.watermarkSize == self.imgSize // self.blockSize)

        encodedImg = np.zeros_like(self.img)
        h, w = self.watermarkSize
        for col in range(w):
            for row in range(h):
                encodedImg[
                    row * self.blockSize: (row + 1) * self.blockSize,
                    col * self.blockSize: (col + 1) * self.blockSize
                ] = self.encodeBlockBit(row, col)
        return encodedImg.astype("uint8")


class Decoder:
    """
    For decoding a watermark from an encoded image.

    Attributes:
    - blockSize: int
        Size of each block for decoding.
    - plane: int
        Bit plane used for decoding.
    """

    def __init__(self, blockSize=2, plane=1):
        """
        Initialize the Decoder with the specified block size and bit plane.

        Parameters:
        - blockSize: int, optional
            Size of each block for decoding. Default is 2.
        - plane: int, optional
            Bit plane used for decoding. Default is 1.
        """
        self.plane = plane - 1
        self.blockSize = blockSize

    def decode(self, img):
        """
        Decode the watermark from the encoded image.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the encoded image.
            If numpy.ndarray, the encoded image itself.

        Returns:
        - numpy.ndarray
            Decoded watermark.
        """
        image = Image(img)
        self.img, self.imgSize = image.image, image.imageSize
        self.watermarkSize = self.imgSize // self.blockSize
        watermark = np.zeros(self.watermarkSize)

        imgPlane = self.img & (1 << self.plane)
        h, w = self.watermarkSize
        for col in range(w):
            for row in range(h):
                imgBlock = imgPlane[
                    row * self.blockSize: (row + 1) * self.blockSize,
                    col * self.blockSize: (col + 1) * self.blockSize
                ]
                encByte = 255 * np.round(imgBlock.mean(), 0)
                watermark[row, col] = encByte
        return watermark.astype("uint8")
