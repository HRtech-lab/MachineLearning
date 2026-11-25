# Images

Edges, gradient descriptors, convolutional classification and segmentation.

| Example | Question or method |
|---|---|
| [Canny Edge Detection](01%20Canny%20Edge%20Detection/canny_edge_detection.ipynb) · [Python](01%20Canny%20Edge%20Detection/canny_edge_detection.py) | How close is a detected edge to the actual boundary of a noisy shape? A synthetic mask supplies the location to within a pixel. |
| [HOG and Linear SVM](02%20HOG%20and%20Linear%20SVM/hog_and_linear_svm.ipynb) · [Python](02%20HOG%20and%20Linear%20SVM/hog_and_linear_svm.py) | Gradient orientations describe the strokes of a digit. A linear margin classifier then works on those local histograms. |
| [Convolutional Digit Classifier](03%20Convolutional%20Digit%20Classifier/convolutional_digit_classifier.ipynb) · [Python](03%20Convolutional%20Digit%20Classifier/convolutional_digit_classifier.py) | Two convolutional layers learn local stroke features directly from the original eight-pixel-wide digits. |
| [U-Net Segmentation](04%20U-Net%20Segmentation/u_net_segmentation.ipynb) · [Python](04%20U-Net%20Segmentation/u_net_segmentation.py) | A small encoder–decoder finds disks and rectangles under uneven lighting. A skip connection returns fine edges to the decoder. |
