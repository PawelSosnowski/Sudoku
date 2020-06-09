import cv2 as cv
import numpy as np
from scipy.signal import find_peaks


class Board:

    def __init__(self, name, image=0, resized=0, eboard=0):
        self.name = name
        self.image = image
        self.resized = resized
        self.extracted_board = eboard

    def find_max_contour(self, contours):
        maxarea = 0
        max_contour = 0
        for con in contours:
            conarea = cv.contourArea(con)
            if conarea > maxarea:
                maxarea = conarea
                max_contour = con
        return max_contour

    def preparation(self):
        ori_img = cv.imread(self.name, 0)
        if ori_img is None:
            raise AttributeError(' Incorrect name of image')
        h = int(ori_img.shape[0] * 0.15)
        w = int(ori_img.shape[1] * 0.15)
        ori_img = cv.resize(ori_img, (w, h), interpolation=cv.INTER_AREA)
        img = np.copy(ori_img)
        img = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
        img = cv.dilate(img, (9, 9))
        img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 4)
        self.image = img
        self.resized = ori_img

    def crop_grid(self):
        img = np.copy(self.image)

        # looks for contours and finds max
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        max_contour = self.find_max_contour(contours)

        mask = np.zeros_like(img)
        cv.drawContours(mask, [max_contour], -1, color=(255, 255, 255), thickness=1)
        # cv.imshow('maska', mask)
        # cv.waitKey(0)

        # converts from cartesian to polar coordinates: angle [deg]
        def to_polar(x, y):
            if (x == 0) and (y >= 0):
                angle = 90
            elif (x == 0) and (y < 0):
                angle = 270
            else:
                angle = np.arctan(y / x) * 180 / np.pi
            if (x < 0) and (y >= 0):
                angle += 180
            if (x < 0) and (y < 0):
                angle += 180
            if (x > 0) and (y < 0):
                angle += 360
            r = np.sqrt(x * x + y * y)
            return angle, r

        x, y = np.where(mask == 255)
        maxx = max(x)
        maxy = max(y)
        minx = min(x)
        miny = min(y)
        center = ((minx + maxx) / 2, (miny + maxy) / 2)

        # finding corners of the grid
        xc, yc = map(lambda i: i - center[0], x), map(lambda i: i - center[1], y)

        polar_points = list(map(to_polar, xc, yc))
        cpolar_points = polar_points.copy()
        cpolar_points.sort()

        d = []
        for xy in cpolar_points:
            d.append(xy[1])

        corners_indices, _ = find_peaks(d, prominence=25)

        corners_polar = [cpolar_points[corners_indices[0]], cpolar_points[corners_indices[1]],
                         cpolar_points[corners_indices[2]], cpolar_points[corners_indices[3]]]

        corners_indices[0] = polar_points.index(corners_polar[0])
        corners_indices[1] = polar_points.index(corners_polar[1])
        corners_indices[2] = polar_points.index(corners_polar[2])
        corners_indices[3] = polar_points.index(corners_polar[3])

        # corners[0] - right_top, corners[1] - left_top, corners[2] - left_down, corners[3] - right_down
        corners = np.array([[x[corners_indices[0]], y[corners_indices[0]]],
                            [x[corners_indices[1]], y[corners_indices[1]]],
                            [x[corners_indices[2]], y[corners_indices[2]]],
                            [x[corners_indices[3]], y[corners_indices[3]]]])

        board_img = np.copy(self.resized)
        # cv.circle(board_img, (corners[0][1], corners[0][0]), 5, 0, -1)
        # cv.circle(board_img, (corners[1][1], corners[1][0]), 5, 0, -1)
        # cv.circle(board_img, (corners[2][1], corners[2][0]), 5, 0, -1)
        # cv.circle(board_img, (corners[3][1], corners[3][0]), 5, 0, -1)
        # cv.imshow('corners', board_img)
        # cv.waitKey(0)

        # Correcting the skewed perspective, real board aspect ratio is 58x63
        def correct_perspective(p_corners):
            new_corners = np.array([[579, 629], [0, 629], [0, 0], [579, 0]])
            crn = np.array([[p_corners[0][1], p_corners[0][0]], [p_corners[1][1], p_corners[1][0]],
                            [p_corners[2][1], p_corners[2][0]], [p_corners[3][1], p_corners[3][0]]])
            h, _ = cv.findHomography(crn, new_corners)

            extracted_board = cv.warpPerspective(board_img, h, (580, 630))
            extracted_board = cv.flip(extracted_board, 0)
            extracted_board = cv.rotate(extracted_board, cv.ROTATE_90_CLOCKWISE)
            return extracted_board

        self.extracted_board = correct_perspective(corners)
        cv.imshow(self.name, self.resized)
        cv.imshow(self.name, self.extracted_board)

    def grid_numbers(self):
        cropped_board = np.copy(self.extracted_board)
        cropped_board = cv.adaptiveThreshold(cropped_board, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 4)
        width = []
        height = []
        cells = []
        step_width = cropped_board.shape[0] // 9
        step_height = cropped_board.shape[1] // 9
        for i in range(0, cropped_board.shape[0]+1, step_width):
            width.append(i)
        for i in range(0, cropped_board.shape[1]+1, step_height):
            height.append(i)

        for i in range(0, (len(width)-1)):
            for j in range(0, (len(height)-1)):
                cells.append(cropped_board[width[i]:width[i+1], height[j]:height[j+1]])
        return cells

    def recognize_digits(self):
        cells = self.grid_numbers()
        board_of_digits = []
        digit_contour = []
        digit_template = []
        for i in range(1, 10):
            template = cv.imread(r'.\templates\number'+str(i)+'.jpg', 0)
            if template is None:
                raise ValueError('uncorrect path to image')
            tp = np.copy(template)
            _, template = cv.threshold(template, 120, 255, 0)
            template = cv.medianBlur(template, 5, cv.BORDER_DEFAULT)
            template = cv.adaptiveThreshold(template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 4)

            contours, _ = cv.findContours(template, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            max_contour = self.find_max_contour(contours)
            digit_contour.append(max_contour)

            masktemp = np.zeros_like(tp)
            for contour in contours:
                cv.drawContours(masktemp, [contour], -1, color=(255, 255, 255), thickness=1)
            # cv.imshow('maska', masktemp)
            # cv.waitKey(0)
            digit_template.append(masktemp)

        def end_points_of_contour(max_contour):
            my = max(max_contour[:, 0, 1])
            mx = max_contour[np.argmax(max_contour[:, 0, 0]), 0, 0]
            ny = min(max_contour[:, 0, 1])
            nx = max_contour[np.argmin(max_contour[:, 0, 0]), 0, 0]

            # mask = np.zeros_like(template)
            # cv.drawContours(mask, [max_contour], -1, color=(255, 255, 255), thickness=1)
            # cv.imshow('maska', mask)
            # cv.waitKey(0)

            # M = cv.moments(mask)
            # cx = int(M["m10"] / M["m00"])
            # cy = int(M["m01"] / M["m00"])
            # cv.circle(mask, (cx, cy), 5, 255, -1)
            # cv.circle(mask, (mx, my), 5, 255, -1)
            # cv.circle(mask, (nx, ny), 5, 255, -1)
            # cv.imshow('centroid', mask)
            # cv.waitKey(0)
            # # gora:{nx,ny} dol:{mx,my} centroid: {cx,cy}, ratio: {cy / abs(ny-my)}')
            # ratio = cy / abs(ny-my)
            return my, mx, ny, nx

        for i in range(81):
            cells[i] = cv.resize(cells[i], (cells[i].shape[0]*3, cells[i].shape[1]*3), interpolation=cv.INTER_AREA)
            _, template = cv.threshold(cells[i], 120, 255, 0)
            template = cv.medianBlur(template, 7, cv.BORDER_DEFAULT)
            template = cv.adaptiveThreshold(template, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 4)
            # cv.imshow('template', template)
            # cv.waitKey(0)
            contours, _ = cv.findContours(template, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            max_contour = self.find_max_contour(contours)

            image_of_max_contour = np.zeros_like(template)
            cv.drawContours(image_of_max_contour, [max_contour], -1, color=(255, 255, 255), thickness=1)

            maxy, maxx, miny, minx = end_points_of_contour(max_contour)
            rows = np.linspace(minx, maxx, num=maxx-minx+1, dtype=np.uint8)
            columns = np.linspace(miny, maxy, num=maxy-miny+1, dtype=np.uint8)

            masktemp = np.zeros_like(template)
            for contour in contours:
                cv.drawContours(masktemp, [contour], -1, color=(255, 255, 255), thickness=1)
            final = np.zeros_like(template)
            final[columns[:, None], rows] = masktemp[columns[:, None], rows]
            # cv.imshow('final', final)
            # cv.waitKey(0)

            score = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0} # Bank of digits

            for digit, contour in enumerate(digit_contour):
                score[digit+1] = cv.matchShapes(max_contour, contour, 3, 0.0)
            is_digit = min(score, key=score.get)
            if score[is_digit] < 1:
                if is_digit in [6, 8, 9]:
                    contours, _ = cv.findContours(template, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
                    if score[is_digit] < 0.08:
                        is_digit = 8
                    else:
                        output = np.zeros_like(template)
                        circles = cv.HoughCircles(final, cv.HOUGH_GRADIENT, 1.6, 50)
                        if circles is not None:
                            circles = np.round(circles[0, :]).astype("int")
                            for (x, y, r) in circles:
                                ratio = abs(y-miny)/abs(maxy-miny)
                                cv.circle(output, (x, y), r, 255, 3)
                                if ratio < 0.4:
                                    is_digit = 9
                                else:
                                    is_digit = 6
                        # cv.imshow('output', output)
                        # cv.waitKey(0)
                if is_digit in [2, 5]:
                    is_digit = 5
                    hough_lines = cv.HoughLines(image_of_max_contour, 1, 2 * np.pi / 180, 40)
                    hough_img = np.zeros_like(template, dtype=np.uint8)
                    if hough_lines is not None:
                        for line in hough_lines:
                            rho, theta = line[0]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            if b == 0 or abs(a/b) > 1:
                                break
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * a)
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * a)
                            cv.line(hough_img, (x1, y1), (x2, y2), 255, 1)
                        indices = hough_img.nonzero()
                        maxy_hough = max(indices[0])
                        miny_hough = min(indices[0])
                        y_of_line = (maxy_hough+miny_hough)/2
                        hough_ratio = (maxy-miny)/(maxy-y_of_line)
                        if hough_ratio > 5:
                            is_digit = 2
                    else:
                        is_digit = 3
                    # cv.imshow('hough', hough_img)
                    # cv.waitKey(0)
                board_of_digits.append(is_digit)
            else:
                board_of_digits.append(0)
        print("TABLICA SUDOKU: ")
        for n in range(0, 81, 9):
            print(board_of_digits[n:n+9])
        cv.waitKey(0)


    def run(self):
        self.preparation()
        self.crop_grid()
        self.recognize_digits()
        self.__del__()

    def __del__(self):
        cv.destroyAllWindows()


if __name__ == '__main__':
    jpg_name = input("Podaj nazwe zdjecia, np. 'sudoku.jpg': ")
    o = Board(jpg_name)
    o.run()
