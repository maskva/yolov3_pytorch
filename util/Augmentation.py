from torchvision import transforms
import cv2 as cv

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Resize(object):
    """ 调整图像大小 """
    def __init__(self, size=(416, 416)):
        self.size = size

    def __call__(self, image, bbox=None, label=None):
        """ 调整图像大小

        Parameters
        ----------
        image: `~np.ndarray` of shape `(H, W, 3)`
            RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            已经归一化的边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签

        Returns
        -------
        image, bbox, label:
            增强后的数据
        """
        return cv.resize(image, self.size), bbox, label

class  Augmentation(object):
    """ Yolo 训练时使用的数据集增强器 """
    def __init__(self, image_size: int) -> None:
        """
        Parameters
        ----------
        image_size: int
            图像缩放后的尺寸
        """
        self.transformers = Compose([
            Resize((image_size, image_size))
        ])

    def __call__(self, image, bbox, label):

        return self.transformers(image, bbox, label)




