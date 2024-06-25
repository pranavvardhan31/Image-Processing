import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# Reading the csv file
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            row = line.strip().split("\t")
            data.append(row[1])
    return np.array(data, dtype=np.float64)

# 1.Q
book1 = read_csv("book1.csv")
# book1 is an NDarray (N-Dimensional array) with floating point and a precision of 64 bits
print(book1)
max_val = np.max(book1)
min_val = np.min(book1)
print(max_val)  # printing the maximum value contained in the book1
print(min_val)  # printing the minimum value contained in the book1

# 2.Q
sorted_array = np.sort(book1)
print("Sorted array for Book1: ", sorted_array)

# 3.Q
reversed_array = sorted_array[::-1]
print("Reversed array for Book1: ", reversed_array)

# 4.Q
book2 = read_csv("book2.csv")  # reading book2.csv
book3 = read_csv("book3.csv")  # reading book3.csv

# mean = sum(book(i))/len(book(i))
mean1 = sum(book1) / len(book1)
mean2 = sum(book2) / len(book2)
mean3 = sum(book3) / len(book3)
mean_list = [mean1, mean2, mean3]
print(mean_list)

# 5.Q
# Function to display an image
def display_image(image, window_name='output'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('a.png', cv2.IMREAD_COLOR)  # for 6.Q
color_image = img
display_image(img)  # displays color image

# 6.Q
array_X = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray Image (color image to gray image)
gray_image = array_X
cv2.imshow('Grayscale Image', array_X)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7.Q
array_Y = np.transpose(array_X)  # Y is transpose of X
start_time = time.time()
array_Z = np.matmul(array_X, array_Y)  # array_Z = (array_X)×(array_Y)
end_time = time.time()  # Use this for 8.Q
print("Resultant Matrix: ")
print(array_Z)

# 8.Q
print(f"Time taken with NumPy: {end_time - start_time} seconds")
start_time = time.time()
array_Z = array_X.dot(array_Y)
end_time = time.time()
print(f"Time taken without NumPy: {end_time - start_time} seconds")

# 9.Q
# Plotting the histogram of pixel intensities for a grayscale image
plt.hist(gray_image.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Pixel Intensity Histogram')
plt.show()

# 10.Q
# Drawing a filled rectangle on the grayscale image
rectangle_image = cv2.rectangle(gray_image, (40, 100), (70, 200), (0, 0, 0), -1)
plt.imshow(rectangle_image, cmap='gray')
plt.axis('off')
plt.show()

# 11.Q
# Binarizing the grayscale image using multiple thresholds
thresholds = [50, 70, 100, 150]
binarized_images = [np.where(gray_image > threshold, 1, 0) for threshold in thresholds]

# 12.Q
# Applying a custom filter kernel on the color image
filter_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
filtered_image = cv2.filter2D(color_image, -1, filter_kernel)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
