import cv2
import numpy as np


def ImageLoader(img):
    """
    Load an image from either a file path or a NumPy array.

    Parameters:
    - img: str or numpy.ndarray
        If str, the file path of the image. If numpy.ndarray, the image itself.

    Returns:
    - Tuple[str, numpy.ndarray]
        A tuple containing the image path (if applicable) and the loaded image as a NumPy array.
    """
    image = img
    imgPath = None

    if type(img) == str:
        imgPath = img
        image = cv2.imread(imgPath, 0)

    image.dtype = np.uint8
    return imgPath, image


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
        self.plane = plane-1
        self.blockSize = blockSize

    def getImage(self, img):
        """
        Load the image to be encoded and extract its dimensions.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the image. If numpy.ndarray, the image itself.
        """
        self.imagePath, self.img = ImageLoader(img)
        self.h, self.w = self.img.shape

    def getWatermark(self, watermark):
        """
        Load the watermark image, resize it, and convert it to binary.

        Parameters:
        - watermark: str or numpy.ndarray
            If str, the file path of the watermark image. If numpy.ndarray, the watermark itself.
        """
        self.watermarkPath, self.watermark = ImageLoader(watermark)

        self.watermark = cv2.resize(
            self.watermark, (self.w//self.blockSize, self.h//self.blockSize))
        _, self.watermark = cv2.threshold(
            self.watermark, 127, 255, cv2.THRESH_BINARY)

        self.h_w, self.w_w = self.watermark.shape

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
        img_block = self.img[row*self.blockSize:(row+1)*self.blockSize,
                             col*self.blockSize:(col+1)*self.blockSize]
        enc_block = enc*np.ones_like(img_block, 'uint8')
        masked_img = img_block & (255 - 1 << self.plane)
        masked_enc = enc_block & (1 << self.plane)
        return masked_img + masked_enc

    def encode(self, img, watermark):
        """
        Encode the watermark into the image.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the image. If numpy.ndarray, the image itself.
        - watermark: str or numpy.ndarray
            If str, the file path of the watermark image. If numpy.ndarray, the watermark itself.

        Returns:
        - numpy.ndarray
            Encoded image.
        """
        self.getImage(img)
        self.getWatermark(watermark)

        assert (self.h_w == self.h//self.blockSize and self.w_w == self.w//self.blockSize), \
            f'Image and watermark shapes of {self.img.shape} and {self.watermark.shape} are incompatible.'

        encodedImg = np.zeros_like(self.img)
        for col in range(self.w_w):
            for row in range(self.h_w):
                encodedImg[row*self.blockSize:(row+1)*self.blockSize,
                           col*self.blockSize:(col+1)*self.blockSize] = self.encodeBlockBit(row, col)
        return encodedImg.astype('uint8')


class Decoder:
    """
    For decoding a watermark from an encoded image using a block-based approach.

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
        self.plane = plane-1
        self.blockSize = blockSize

    def getImage(self, img):
        """
        Load the image to be decoded and extract its dimensions.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the image. If numpy.ndarray, the image itself.
        """
        self.imagePath, self.img = ImageLoader(img)
        self.h, self.w = self.img.shape
        self.h_w, self.w_w = self.h//self.blockSize, self.w//self.blockSize

    def decode(self, img):
        """
        Decode the watermark from the encoded image.

        Parameters:
        - img: str or numpy.ndarray
            If str, the file path of the encoded image. If numpy.ndarray, the encoded image itself.

        Returns:
        - numpy.ndarray
            Decoded watermark.
        """
        self.getImage(img)
        watermark = np.zeros((self.h_w, self.w_w))

        img_plane = self.img & (1 << self.plane)
        for col in range(self.w_w):
            for row in range(self.h_w):
                img_block = img_plane[row*self.blockSize:(row+1)*self.blockSize,
                                      col*self.blockSize:(col+1)*self.blockSize]
                enc_byte = 255 * np.round(img_block.mean(), 0)
                watermark[row, col] = enc_byte
        return watermark.astype('uint8')
