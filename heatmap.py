import os, numpy, PIL
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import sys

np.set_printoptions(threshold=sys.maxsize)


def test1():
    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(os.path.join(path, imlist[0])).size
    N = len(imlist)

    # Create a numpy array of floats to store the average (assume RGB images)
    arr = numpy.zeros((h, w, 3), dtype=numpy.float32)

    # print(arr)

    # Build up average pixel intensities, casting each image as an array of floats
    for i, im in enumerate(imlist):
        imarr = numpy.array(Image.open(os.path.join(path, im)), dtype=numpy.float32)
        print(imarr)
        arr = arr + imarr / N

    # Round values in array and cast as 8-bit integer
    arr = numpy.array(numpy.round(arr), dtype=numpy.uint8)

    # Generate, save and preview final image
    out = Image.fromarray(arr, mode="RGB")
    out.save("Average.png")
    out.show()


def test2(day_folder, imlist):
    def process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 25)
        img_canny = cv2.Canny(img_blur, 5, 50)
        kernel = np.ones((3, 3))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        return img_erode

    def get_contours(img, img_original):
        img_contours = img_original.copy()
        contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), -1)
        # If you want to omit smaller contours, loop through the detected contours, and only draw them on the image if they are at least a specific area. Don't forget to remove the line above if you choose the below block of code.
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), -1)

        return img_contours

    folder_path = os.path.join(path, day_folder)
    img1 = cv2.imread(os.path.join(folder_path, imlist[0]))
    img2 = cv2.imread(os.path.join(folder_path, imlist[1]))
    heat_map = np.zeros(img1.shape[:-1])
    heat_map_std = np.zeros(img1.shape[:-1])
    for img_path in tqdm(imlist[2:]):
        diff = cv2.absdiff(img1, img2)
        # img_contours = get_contours(process(diff), img1)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype('uint8')
        # ret2, th2 = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, th2 = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        heat_map[(th2 == 255)] += 3
        # heat_map[np    .all(img_contours == [0, 255, 0], 2)] += 3  # The 3 can be tweaked depending on how fast you want the colors to respond
        # heat_map[np.any(img_contours != [0, 255, 0], 2)] -= 3
        # heat_map[heat_map < 0] = 0
        # heat_map[heat_map > 255] = 255
        # heat_map[:250,2000:] = 0
        heat_map_std = ((heat_map - heat_map.min()) * (1 / (heat_map.max() - heat_map.min()) * 255)).astype('uint8')
        # print(heat_map_std)
        img_mapped = cv2.applyColorMap(heat_map_std.astype('uint8'), cv2.COLORMAP_JET)

        #    img1[heat_map > 160] = img_mapped[heat_map > 160] Use this line to draw the heat map on the original video at a specific temperature range. For this it's where ever the temperature is above 160 (min is 0 and max is 255)


        """cv2.namedWindow("Heat Map", cv2.WINDOW_NORMAL)
        # cv2.imshow("Original", img1)
        img_stacked = np.hstack((img_mapped, img2))
        cv2.imshow("Heat Map", img_stacked)
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        cv2.namedWindow("th2", cv2.WINDOW_NORMAL)
        cv2.imshow("th2", th2)"""

        img1 = img2
        img2 = cv2.imread(os.path.join(path, day_folder, img_path))

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.imwrite(os.path.join(path, day_folder, f"heatmap_{day_folder}.jpg"), img_mapped)


def make_smaller():
    for i, img_path in enumerate(imlist):
        print(f"image {i} / {len(imlist)}")
        img = cv2.imread(os.path.join(path, img_path))
        (w, h, _) = img.shape

        img_small = cv2.resize(img, (int(h / 4), int(w / 4)), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(os.path.join(new_path, img_path), img_small)



# make_smaller()

if __name__ == "__main__":
    # Access all PNG files in directory
    path = r"C:\Users\fwern\Desktop\Fish_adjust"

    for day_folder in os.listdir(path):
        imlist=os.listdir(os.path.join(path, day_folder))
        test2(day_folder, imlist)