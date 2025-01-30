import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # Stop if camera fails

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        continue  # Skip if no contours

    # Find the largest contour (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)

    # Get convex hull (indices for convexity defects)
    hull = cv2.convexHull(max_contour, returnPoints=False)

    # Find convexity defects
    if len(hull) > 3:
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            finger_count = 0
            thumb_closed = False

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Find the angle between fingers
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b)) * 57

                if angle < 90:  # Valid finger gap
                    finger_count += 1
                    cv2.circle(img, far, 5, [0, 0, 255], -1)

                # Detect if the thumb is closed (position check)
                if far[0] < img.shape[1] // 3:  # Thumb is usually on the left side
                    thumb_closed = True

            # Draw contours and hull
            cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(img, [cv2.convexHull(max_contour)], -1, (255, 0, 255), 2)

            # Check if exactly 4 fingers are raised & thumb is closed
            if finger_count == 4 and thumb_closed:
                cv2.putText(img, "DANGER ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Input", img)

    if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
