import os
import imageio
import skimage
import numpy as np
import utils
from sys import exit

class DsbDataset(utils.Dataset):

    def load_dataset(self, ids, directory):
        """Initialize instance variables
        1. Add new class to **self.class_info**, in this case, only have 1 more class which is the nuclei.
           The first class is always the background which is already added by super.
        2. Add all information of dataset to **self.image_info**
        """
        self.add_class("dsb", 1, "nuclei")
        for i, id in enumerate(ids):
            image_dir = os.path.join(directory, id)
            self.add_image("dsb", image_id=i, path=image_dir)

    def load_image(self, image_id, non_zero=None):
        """see doc in super
        """
        info = self.image_info[image_id]
        path = info['path']
        image_name = os.listdir(os.path.join(path, 'images'))
        image_path = os.path.join(path, 'images', image_name[0])
        image = imageio.imread(image_path)
        if image.shape[2] != 3:
            image = image[:,:,:3]
        image = self.preprocess(image)
        image = image.astype('float32')
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Get mask matrix(composed with only 0 and 1) for a specific image_id
        Note that this function also handled the occlusions. That means all of the masks are not
        overlapped. It's the same as cutting a single paper by many mask.
        Returns:
            (mask, class_ids)
            mask is a [W, H, number_nuclei] matrix, in fact it should be masks
            class_ids is for every mask's class id, in this case, it's all 1(nuclei class)
        """
        info = self.image_info[image_id]
        path = info['path']
        mask_dir = os.path.join(path, 'masks')
        mask_names = os.listdir(mask_dir)
        count = len(mask_names)
        mask = []
        for i, el in enumerate(mask_names):
            msk_path = os.path.join(mask_dir, el)
            msk = imageio.imread(msk_path)
            if np.sum(msk) == 0:
                print('invalid mask')
                continue
            msk = msk.astype('float32')/255.
            mask.append(msk)
        mask = np.asarray(mask)
        mask[mask > 0.] = 1.
        mask = np.transpose(mask, (1,2,0))
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        count = mask.shape[2]
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = [self.class_names.index('nuclei') for s in range(count)]
        class_ids = np.asarray(class_ids)
        return mask, class_ids.astype(np.int32)

    def preprocess(self, img):
        """8bits image, actually image in grayscale
        """
        gray = skimage.color.rgb2gray(img.astype('uint8'))
        img = skimage.color.gray2rgb(gray)
        img *= 255.
        return img
