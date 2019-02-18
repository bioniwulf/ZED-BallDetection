import cv2
import pyzed.sl as sl
import numpy as np
import time
import math
import os
from functools import *

# Радиус мячика, мм
ball_world_radius = 33.75

# Коэфициент на который будет уменьшен кадр для последующего распознавания
cv_ratio = 1

# Расстояние от среднего цвета мячика
threshold = 50

# Положение курсора
mouse_x, mouse_y = -1, -1
cursor_is_changed = False

font = cv2.FONT_HERSHEY_SIMPLEX
upLeftCornerOfText = (20, 20)
fontScale = 1
fontColor = (0, 255, 255)
lineType = 1

# Обработка событий с мышки


def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, cursor_is_changed
    # При двойном клике получаем текущие положение курсора
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_x, mouse_y = x, y
        cursor_is_changed = True

# Подготовка серошкального изображения с выделением мячика по заранее узнанному цвету


def prepeare_color_image(image_color, ball_color_bgr, threshold):
    # Перевод BGR изображения в HSV
    img_color_hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)

    # перевод цвета мячика в hsv
    ball_color_hsv = cv2.cvtColor(
        np.uint8([[ball_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # формирование нижней и верхней границы для последующего формирования маски
    lower_boader = np.array([ball_color_hsv[0] - threshold / 2,
                             ball_color_hsv[1] - threshold, ball_color_hsv[2] - threshold])
    upper_boader = np.array([ball_color_hsv[0] + threshold / 2,
                             ball_color_hsv[1] + threshold, ball_color_hsv[2] + threshold])
    # lower_boader = np.array([ball_color_hsv[0] - threshold / 2, 0, 0])
    # upper_boader = np.array([ball_color_hsv[0] + threshold / 2, 255, 255])

    # выделение изображения по верхней и нижней цветовой границам
    mask = cv2.inRange(img_color_hsv, lower_boader, upper_boader)

    # обравка цветного изображения по сформированной маске
    selected_image = cv2.bitwise_and(img_color_hsv, img_color_hsv, mask=mask)
    return selected_image

# формирование серошкального изображения


def prepeare_gray_image(image_color):
    image_color_rgb = cv2.cvtColor(image_color, cv2.COLOR_HSV2RGB)
    image_color_gray = cv2.cvtColor(image_color_rgb, cv2.COLOR_RGB2GRAY)
    return image_color_gray

# Дополнительная обработка изображения


def processing_image(image_gray):

    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)
    image_gray = cv2.addWeighted(image_gray, 1.5, image_gray, 1.5, 0)
    return image_gray

# Поиск окнужности на сером изображении по контуру


def find_circle_contour(img_gray):
    # Поиск всех контуров на изображении
    _, contours, hierarchy = cv2.findContours(
        img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Сортируем контуры по похожести на круг начиная с самых похожих
    # Пока как-то не очень корректно работает
    # contours.sort(key=get_circle_ratio, reverse=True)

    # Удаляем 30% самых непохожих контуров
    # contours = contours[: int(len(contours) * 0.7)]
    # circle = None
    contour = None
    centers = []

    if(len(contours) != 0):
        # Если ещё остались контуры то сортируем их по радиусу начинаю с большого
        contours.sort(key=get_circle_radius, reverse=True)
        # # Считаем что самый первых это и есть наш контур
        # circle = cv2.minEnclosingCircle(contours[0])

        for contour in contours:
            # compute the center of the contour
            M = cv2.moments(contour)
            center_x = M["m10"] / M["m00"]
            center_y = M["m01"] / M["m00"]
            centers.append((center_x, center_y))
        contour = contours[0]

    return centers, contour

# Поиск окружности на сером изображении методом Хафа


def find_circle_hough(img_gray):
    # используем детектор хафа для поиска окружности
    # возможно будет не очень стабильно работать на большом расстоянии
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1.0, 100,
                               param1=100, param2=20, minRadius=0, maxRadius=0)

    if circles is not None:
        circle = circles[0][0]
        x, y, r = circle[0], circle[1], circle[2]
        circle = (x, y, r)
    else:
        circle = None  # Если контура нет то возвращаем None
    return circle

# Ограничение точки по нижнему и верхнему пределу


def saturation(point, lower_limit, upper_limit):
    for axis_val in point:
        axis_val = axis_val if axis_val > lower_limit else lower_limit
        axis_val = axis_val if axis_val < upper_limit else upper_limit

    return point

def saturation_point(point, shape):
    point[0] = point[0] if point[0] > 0 else 0
    point[0] = point[0] if point[0] < shape[0] else shape[0]

    point[1] = point[1] if point[1] > 0 else 0
    point[1] = point[1] if point[1] < shape[1] else shape[1]

    return point
# Вычисление средней отфильтрованной глубины


def get_mean_depth(mat_measure_depth, center_point, radius):
    # Преобразование центральной точки в int
    center_point = center_point.astype(int)

    # Подготавливаем границы квадрата для взятие среза по глубине
    horizontal_boadrers = (center_point[0] - radius, center_point[0] + radius)
    horizontal_boadrers = saturation(
        horizontal_boadrers, 0, mat_measure_depth.get_width())

    vertical_boadrers = (center_point[1] - radius, center_point[1] + radius)
    vertical_boadrers = saturation(
        vertical_boadrers, 0, mat_measure_depth.get_height())

    # берем двумерный срез массива вокруг центральной точки
    depth_points = mat_measure_depth.get_data()[vertical_boadrers[0]: vertical_boadrers[1],
                                                horizontal_boadrers[0]: horizontal_boadrers[1]]

    # Производим фильтрацию значений на nan, оставлям только то что валидно
    # заодно это выражение преобразует 2D в 1D
    depth_points = depth_points[depth_points == depth_points.astype(float)]

    # если после этого оказалось что массив пуст, то возвращаем None
    if depth_points.size == 0:
        return None

    # В противном случае меняем тип массива, ибо до этого он был object
    depth_points = depth_points.astype(float)

    # Сортируем массив в порядке возрастания
    depth_points = np.sort(depth_points)

    # И возвращаем среднее число
    depth_value = mat_measure_depth.get_value(center_point[1], center_point[0])[
        1]  # depth_points[int(depth_points.size / 2)]
    if np.isnan(depth_value) or np.isinf(depth_value):
        return None
    else:
        return depth_value

# Определение расстояния зная диаметр реального мячика


def get_depth_mono(image_radius, camera_parameters):
    depth = camera_parameters.left_cam.fy * ball_world_radius / image_radius
    return depth

def get_depth_new(center_left, center_right, camera_parameters):
    (x_left, y_left) = center_left
    (x_right, y_right) = center_right
    stereo_difference = x_left - x_right

    # Стереобаза между камерами. Берется из матрицы трансляции правой камеры
    # относительно левой
    camera_stereobase = camera_parameters.T[0]
    if(abs(stereo_difference) > 0.01):
        depth = camera_parameters.left_cam.fy * camera_stereobase / stereo_difference
    else:
        depth = None
    return depth

# Формирование пространственных координат точки XYZ
# Работаем с левой камерой
# Считаем ось Z направлена от камеры вперед
# Ось X направлена вверх
# Ось Y направлена вправо


def get_coordinate_xyz(coordinate_image, depth_world, camera_parameters):
    if depth_world is None:
        return None

    coordinate_image_y = coordinate_image[0]
    coordinate_image_x = coordinate_image[1]

    principal_point_x = camera_parameters.left_cam.cy
    principal_point_y = camera_parameters.left_cam.cx

    focal_length_x = camera_parameters.left_cam.fy
    focal_length_y = camera_parameters.left_cam.fx

    coordinate_world_z = depth_world
    coordinate_world_y = depth_world * \
        (coordinate_image_y - principal_point_y) / focal_length_y
    coordinate_world_x = -depth_world * \
        (coordinate_image_x - principal_point_x) / focal_length_x

    return np.array([coordinate_world_x, coordinate_world_y, coordinate_world_z])

# Инициализация камеры


def config_camera():
    # Настроечные параметры камеры
    init = sl.InitParameters()
    # Экземпляр камеры
    cam = sl.Camera()

    # Предварительная настройка камеры
    # Используемая система координат
    init.coordinate_system = sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP
    # Типы формирования карты глубины
    # DEPTH_MODE_QUALITY DEPTH_MODE_PERFORMANCE DEPTH_MODE_MEDIUM DEPTH_MODE_ULTRA DEPTH_MODE_NONE
    init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_NONE
    # Используем мм в качестве системных единиц
    init.coordinate_units = sl.UNIT.UNIT_MILLIMETER
    # Минимальное расстояние для распознавания глубины 300 мм
    init.depth_minimum_distance = 300
    # Разрешение каqмеры
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    # частота кадров
    init.camera_fps = 60
    # Стабилизация карты глубины
    init.depth_stabilization = False

    # Пытаемся открыть камеру
    print("Trying to open camera")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return (None, None)
    else:
        print("Camera opened successfully")

    runtime_settings = sl.RuntimeParameters()
    # SENSING_MODE_STANDARD SENSING_MODE_FILL
    runtime_settings.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
    runtime_settings.measure3D_reference_frame = sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA

    # Настройка камеры
    # Маленькая экспозиция что бы не было размытия
    cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, 100)

    # Максимальная дальность работы стерео
    cam.set_depth_max_range_value(10000)
    # Возвращаем экземляр открытой камеры
    return cam, runtime_settings


def get_circle_radius(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return radius


def get_circle_ratio(contour):
    return abs(cv2.arcLength(contour, True) - 2 * math.pi * get_circle_radius(contour))

def convert_to_global(center, roi):
    x, y = center
    x += roi[0][1]
    y += roi[0][0]
    return (x, y)

def draw_debug_info(image, title, center, depth, coordinate, velocity):
    cv2.putText(image, "Depth {1}: {0} mm.".format(round(depth, 0), title),
                        (20, 30), font, fontScale, fontColor, lineType)
    cv2.putText(image, "Coordinate {3}: ({0}, {1}, {2}) mm".format(coordinate[0].round(1),
                                 coordinate[1].round(1), coordinate[2].round(1), title),
                (20, 90), font, fontScale, fontColor, lineType)

    x, y = center
    x = x * cv_ratio
    y = y * cv_ratio
    ptr_start = (int(x), int(y))
    ptr_end = (
        int(x) + int(velocity[1] / 5), int(y) + int(-velocity[0] / 5))
    cv2.arrowedLine(image, ptr_start, ptr_end, (0, 0, 255), 5)

def find_circle(image, color, roi):
    # # Уменьшаем кадр в ratio раз что бы быстрее работали алгоритмы распознавания
    # image_resize = cv2.resize(image, (0, 0), fx=1 / cv_ratio, fy=1 / cv_ratio)

    # Выделяем необходимую область интереса
    image_roi = image[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]

    # print("Image size: {0}".format(image.shape))
    # print("Image ROI size: {0}".format(image_roi.shape))
    # Подготовка цветного изображения с выделением мячика по цвету
    image_color_selected = prepeare_color_image(image_roi, color, threshold)

    # Подготовка серошкального изображения
    image_gray_selected = prepeare_gray_image(image_color_selected)

    # Дополнительная обработка серошкального изображения
    image_gray = processing_image(image_gray_selected)

    # Ищем мяч на изображении
    # find_circle_hough(image_gray)
    centers, det_contour = find_circle_contour(image_gray)

    # Сортируем по схожести центральной точки с заданным
    # цветом
    center = None
    if len(centers) is not 0:
        center = centers[0]


    if center is not None:
        center = convert_to_global(center, roi)
        # convert_to_global(det_contour, roi)
        (x, y) = center

        left_upper_point = saturation_point([int(y - 150), int(x - 150)], image.shape)
        right_lower_point = saturation_point([int(y + 150), int(x + 150)], image.shape)

        roi = [left_upper_point, right_lower_point]

    return center, roi

def main():
    global cursor_is_changed
    # Предустановленный
    ball_clr = [142, 254, 253]

    # Инициализация камеры, если не получилось то закрываем приложение
    cam, runtime_settings = config_camera()
    if cam is None:
        exit()

    # Настройка события связанного с мышкой
    cv2.namedWindow('ZED')
    cv2.setMouseCallback('ZED', mouse_callback)

    # Координаты прямоугольника где последний раз видели мячик
    lastsee_roi = np.array([[0, 0], [0, 0]])
    lastsee_roi_full = np.array([[0, 0], [0, 0]])

    # ZED матрица изображения с камеры
    mat_image_color_left = sl.Mat()  # Цветное изображение с левой камеры
    mat_image_color_right = sl.Mat()  # Цветное изображение с правой камеры
    mat_image_depth = sl.Mat()  # Серое изображение глубины

    # ZED матрица содержащя измерения
    mat_measure_depth = sl.Mat()  # Измерения глубины пикселей
    mat_measure_coord = sl.Mat()  # Измерения координат пикселей

    # Начальная отметка времени
    start_time = time.time()
    previous_time = time.time()

    save_path = "image/"

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    file = open(save_path + "data.txt", "w+")
    write_header = "Time, ms;"
    write_header += "Coordinate mono X, mm;Coordinate mono Y, mm;Coordinate mono Z, mm;"
    write_header += "Velocity mono X, mm/s;Velocity mono Y, mm/s;Velocity mono Z, mm/s;"
    write_header += "Coordinate stereo X, mm;Coordinate stereo Y, mm;Coordinate stereo Z, mm;"
    write_header += "Velocity stereo X, mm/s;Velocity stereo Y, mm/s;Velocity stereo Z, mm/s;"
    file.write(write_header + "\n")

    # Первичная (предыдущее) положение мячика
    previous_coordinate_mono = np.array([0, 0, 0])
    previous_coordinate_stereo = np.array([0, 0, 0])

    # Бесконечный цикл пока не будет нажата кнопка q
    key = ''
    while key != 113:

        # Формируем кадр с камеры
        err = cam.grab(runtime_settings)

        # Если кадр взять не получилось, просто пробуем заново
        if err != sl.ERROR_CODE.SUCCESS:
            continue

        # Если успешно то загружаем матрицу содержащую кадр
        # Чем меньше загрузок картинок тем выше FSP
        # цветное изображение с левой камеры
        cam.retrieve_image(mat_image_color_left, sl.VIEW.VIEW_LEFT)
        cam.retrieve_image(mat_image_color_right, sl.VIEW.VIEW_RIGHT)
        
        # Если ROI не проинициализирована то выдаем полный кадр
        if lastsee_roi_full[0][0] == lastsee_roi_full[1][0] and lastsee_roi_full[0][1] == lastsee_roi_full[1][1]:
            lastsee_roi_full[1][0] = mat_image_color_left.get_height()
            lastsee_roi_full[1][1] = mat_image_color_left.get_width()
            lastsee_roi = lastsee_roi_full

        # cam.retrieve_image(mat_image_depth, sl.VIEW.VIEW_DEPTH)  # Серое изображение глубины

        # # Загружаем карты измерений
        # Чем больше загрузок карт тем ниже FPS - особенно это заметно на sl.MEASURE.MEASURE_XYZ
        # cam.retrieve_measure(mat_measure_depth, sl.MEASURE.MEASURE_DEPTH, sl.MEM.MEM_CPU, int(mat_image_color_left.get_width() / cv_ratio),
                             # int(mat_image_color_left.get_height() / cv_ratio))
        # cam.retrieve_measure(mat_measure_coord, sl.MEASURE.MEASURE_XYZ)

        # # Формирование кадра для работы в openCV
        image_color_left = mat_image_color_left.get_data()
        image_color_right = mat_image_color_right.get_data()

        # Автоподстройка цвета мячика по двойному нажатию на цвет мячика на рисунке
        if cursor_is_changed:
            ball_clr = image_color_left[mouse_y][mouse_x][:3]
            print("New color: {0}".format(ball_clr))
            cursor_is_changed = False

        # print("Left frame. Last see ROI: {0}".format(lastsee_roi))

        circle_left, lastsee_roi = find_circle(image_color_left, ball_clr, lastsee_roi)
        if circle_left is None:
            print("Left frame. Fault to detect in ROI. Will try: {0}".format(lastsee_roi_full))
            circle_left, lastsee_roi = find_circle(image_color_left, ball_clr, lastsee_roi_full)

        print("Right frame. Last see ROI: {0}".format(lastsee_roi))

        circle_right, lastsee_roi = find_circle(image_color_right, ball_clr, lastsee_roi)
        if circle_right is None:
            print("Right frame. Fault to detect in ROI. Will try: {0}".format(lastsee_roi_full))
            circle_right, lastsee_roi = find_circle(image_color_right, ball_clr, lastsee_roi_full)

        print("End. Last see ROI: {0}".format(lastsee_roi))

        depth_stereo = None
        depth_mono = None
        coordinate_stereo = None
        coordinate_mono = None

        # Если определили круг то ищем глубину кадра и координаты
        if circle_left is not None and circle_right is not None:
            x, y = circle_left
            # Т.к. карту глубины мы тоже брали урезанную, то тут коодинаты к реальным не приводим
            depth_stereo = get_mean_depth(
                mat_measure_depth, np.array([x, y]), 2)

            # Не забываем обратно привести координаты к реальным размерам
            x = x * cv_ratio
            y = y * cv_ratio

            # depth_mono = get_depth_mono(
                # radius, cam.get_camera_information().calibration_parameters)
            depth_mono = get_depth_new(circle_left, circle_right, cam.get_camera_information().calibration_parameters)

            coordinate_stereo = get_coordinate_xyz(np.array([x, y]), depth_stereo,
                                                   cam.get_camera_information().calibration_parameters)

            coordinate_mono = get_coordinate_xyz(np.array([x, y]), depth_mono,
                                                 cam.get_camera_information().calibration_parameters)

            cv2.circle(image_color_left, (int(x), int(y)),
                       int(10), (0, 255, 0), 2)
            cv2.circle(image_color_left, (int(x), int(y)), 2, (0, 0, 255), 3)

            # cv2.drawContours(image_color_left, [contour], -1, (0, 255, 0), 2)

        # Фиксируем текущее время и разницу времени
        current_time = time.time()
        delta_time = current_time - previous_time
        previous_time = time.time()

        if depth_mono is not None:
            velocity_mono = (coordinate_mono -
                             previous_coordinate_mono) / delta_time
            previous_coordinate_mono = coordinate_mono

        if depth_stereo is not None:
            velocity_stereo = (coordinate_stereo -
                               previous_coordinate_stereo) / delta_time
            previous_coordinate_stereo = cqoordinate_stereo

        # Печать отладочной информации на экран
        if depth_stereo is not None:
            draw_debug_info(image_color_left, "stereo", circle_left, depth_stereo, coordinate_stereo, velocity_stereo)
        elif depth_mono is not None:
            draw_debug_info(image_color_left, "mono", circle_left, depth_mono, coordinate_mono, velocity_mono)

        # Фиксируем FPS как величину обратную прошедшему времени c предыдущего кадра
        cv2.putText(image_color_left, "FPS: {0}".format(round(1 / delta_time, 1)),
                    (image_color_left.shape[1] - 200, 60), font, fontScale, fontColor, lineType)

        print(round(1 / delta_time, 1))
        # выводим изображения на экран
        cv2.imshow("ZED", image_color_left)

        cur_dtime_ms = int((time.time() - start_time) * 1000)

        if circle_left is not None and circle_right is not None:
            image_resize = cv2.resize(image_color_left, (0, 0), fx=1 / 2, fy=1 / 2)
            cv2.imwrite(save_path + "result_image-{0}ms".format(cur_dtime_ms) + ".bmp", image_resize)
        # cv2.imwrite(save_path + "src_image-{0}ms".format(cur_dtime_ms) + ".jpg", image_color_src)
        # cv2.imwrite(save_path + "src_image_depth-{0}ms".format(cur_dtime_ms) + ".jpg", mat_image_depth.get_data())

        # Если данных нет, то пишем нули что бы нормально сформировать табличные данные
        if depth_stereo is None:
            depth_stereo = 0.0
            coordinate_stereo = [0.0, 0.0, 0.0]
            velocity_stereo = [0.0, 0.0, 0.0]

        if depth_mono is None:
            depth_mono = 0.0
            coordinate_mono = [0.0, 0.0, 0.0]
            velocity_mono = [0.0, 0.0, 0.0]

        # Округляем до первого знака после запятой, избыточно конечно, но ничего страшного
        coordinate_mono = [round(el, 1) for el in coordinate_mono]
        velocity_mono = [round(el, 1) for el in velocity_mono]
        coordinate_stereo = [round(el, 1) for el in coordinate_stereo]
        velocity_stereo = [round(el, 1) for el in velocity_stereo]

        # Запись данных в лог файл
        if circle_left is not None and circle_right is not None:
            write_string = "{0};".format(cur_dtime_ms)
            write_string += "{0};{1};{2};".format(
                coordinate_mono[0], coordinate_mono[1], coordinate_mono[2])
            write_string += "{0};{1};{2};".format(
                velocity_mono[0], velocity_mono[1], velocity_mono[2])
            write_string += "{0};{1};{2};".format(
                coordinate_stereo[0], coordinate_stereo[1], coordinate_stereo[2])
            write_string += "{0};{1};{2};".format(
                velocity_stereo[0], velocity_stereo[1], velocity_stereo[2])
            file.write(write_string + "\n")

        # Ожидаем нажатия кнопки что бы можно было закрыть приложение с клавиатуры
        key = cv2.waitKey(1)

    if not file.closed:
        file.close()

    # Если выходим, то закрываем камеру и убиваем окно
    cv2.destroyAllWindows()
    cam.close()
    print("Finish")


if __name__ == "__main__":
    main()
