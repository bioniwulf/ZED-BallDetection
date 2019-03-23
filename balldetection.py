import cv2
import pyzed.sl as sl
import numpy as np
import time
import json
import os

from math import pi, sqrt, cos, sin
from operator import itemgetter


class DetectMode:
    Real = "Real"
    Simulation = "Simulation"


class CenterCalculationMode:
    ByMoments = "ByMoments"
    ByColors = "ByColors"


class BallDetection():

    def __init__(self, conf_file, detect_mode):
        # Файл конфигурации
        self.config = None
        self.jsonFile = None
        self.ConfigName = conf_file
        # Настроечные параметры при инициализации камеры
        self.detect_mode = detect_mode
        # Инициализационные параметры
        self.init_params = sl.InitParameters()
        # Экземпляр камеры
        self.camera = sl.Camera()
        # Настроечные параметры изменяемые в рантайме
        self.runtime_settings = None
        # Левое цветное изображение
        self.image_color_left = None
        # Правое цветное изображение
        self.image_color_right = None
        # Последний статус захвата изображения
        self.is_grab_success = False
        # Расчет центра мяча по цвету или по моментам
        self.center_calculation = CenterCalculationMode.ByMoments
        # Калибровочные параметры камеры
        self.calibration_parameters = None
        # Начальное время
        self.start_time = time.time()
        # Предыдущее время для расчета FPS
        self.previous_time = 0
        # Предыдущее время для расчета скорости
        self.previous_detect_time = 0
        # Последняя разница времени
        self.delta_time = 0
        # Последняя разница времени распознавания
        self.delta_detect_time = 0
        # Последняя задетектированная скорость
        self.velocity = [0.0, 0.0, 0.0]
        # Последнее зафиксированное положение в СК камеры
        self.last_position_camera = [0.0, 0.0, 0.0]
        # Последнее зафиксированное положение в СК робота
        self.last_position_robot = [0.0, 0.0, 0.0]
        # Последняя валидность данных
        self.validity = False
        # Путь к папке с логами
        self.data_path = None
        # Открытый файл с точками траектории в СК камеры
        self.log_file = None
        # Последнее мячика на левом кадре
        self.ball_point_left = (0, 0)
        
        # Загрузка файла конфигурации
        self.jsonFile = open(self.ConfigName, "r", encoding='utf-8')
        self.config = json.load(self.jsonFile)
            

        # Инициализация лога
        self.init_log()

        # Загрузка коэфициентов матрицы перехода из СК камеры в СК робота
        self.calib_data = np.array(self.config["Robot"]["transform_data"])
        self.tranform_matrix = self.create_transform_matrix(self.calib_data)
        # Цвет мяча в BGR
        self.ball_color = self.config["Detection"]["BallColor"]

        # Инициализация камеры
        if self.detect_mode is DetectMode.Real:
            # Загрузка параметров перед инициализацией камеры
            self.camera_initial_config()

            # Пытаемся открыть камеру
            status = self.camera.open(self.init_params)
            print(status)
            if status != sl.ERROR_CODE.SUCCESS:
                raise OSError(status)

            # Загрузка рантайм параметров камеры (которые должны быть загружены после инициализации)
            self.camera_runtime_config()
            self.calibration_parameters = self.camera.get_camera_information().calibration_parameters

    def __del__(self):
        if self.camera.is_opened():
            self.camera.close()

        if self.log_file is not None:
            self.log_file.close()

    # Настройка камеры
    def camera_initial_config(self):
        # Предварительная настройка камеры
        # Используемая система координат
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP
        # Типы формирования карты глубины
        # DEPTH_MODE_QUALITY DEPTH_MODE_PERFORMANCE DEPTH_MODE_MEDIUM DEPTH_MODE_ULTRA DEPTH_MODE_NONE
        self.init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_NONE
        # Используем мм в качестве системных единиц
        self.init_params.coordinate_units = sl.UNIT.UNIT_MILLIMETER
        # Минимальное расстояние для распознавания глубины 300 мм
        self.init_params.depth_minimum_distance = 300
        # Разрешение каqмеры
        self.init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
        # частота кадров
        self.init_params.camera_fps = 60
        # Стабилизация карты глубины
        self.init_params.depth_stabilization = False

    # Настройки камеры после её открытия
    def camera_runtime_config(self):
        self.runtime_settings = sl.RuntimeParameters()
        # Тип формирования карты глубины
        self.runtime_settings.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD
        # СК вычисляемых точек
        self.runtime_settings.measure3D_reference_frame = sl.REFERENCE_FRAME.REFERENCE_FRAME_CAMERA
        # Максимальная дальность работы стерео
        self.camera.set_depth_max_range_value(10000)
        # Настройка камеры
        # Маленькая экспозиция что бы не было размытия
        # cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, 70)
        # cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, 61)
        # cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_AUTO_WHITEBALANCE, False)
        # cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE, 2800)

    def init_log(self):
        path_name_base = self.config["Detection"]["LogDataPath"]
        path_name_modificator = 0

        # Если ничего не логируется то и нафиг папку создавать
        if (self.config["Detection"]["TrajectoryLogging"] or\
            self.config["Detection"]["ImageRawLogging"] or\
            self.config["Detection"]["ImageRecLogging"]) is False:
            print("false")
            return

        # Что бы не было перезаписи, созданем новую папку с инкрементным номером
        while(os.path.isdir(path_name_base + str(path_name_modificator))):
            path_name_modificator += 1

        self.data_path = path_name_base + str(path_name_modificator)
        os.makedirs(self.data_path)
        print("Data folder: {}".format(self.data_path))

        # Создаем файл для записи точек если мы логируем точки
        if self.config["Detection"]["TrajectoryLogging"]:
            self.log_file = open(self.data_path + "/trajectory_camera.txt", "w+")
            write_header = "Time,ms;"
            write_header += "RobotX,mm;RobotY,mm;RobotZ,mm;"
            self.log_file.write(write_header + "\n")

    # Получение пары кадров
    def grab_stereo(self):
        # ZED матрица изображения с камеры
        mat_image_color_left = sl.Mat()  # Цветное изображение с левой камеры
        mat_image_color_right = sl.Mat()  # Цветное изображение с правой камеры

        # Формируем кадр с камеры
        err = self.camera.grab(self.runtime_settings)
        self.is_grab_success = err == sl.ERROR_CODE.SUCCESS

        # Если кадр взять не получилось, выкидываем ошибку
        if self.is_grab_success is not True:
            raise ValueError(err)

        # Если успешно то загружаем матрицу содержащую кадр
        # Чем меньше загрузок картинок тем выше FSP
        # цветное изображение с левой камеры
        self.camera.retrieve_image(mat_image_color_left, sl.VIEW.VIEW_LEFT)
        self.camera.retrieve_image(mat_image_color_right, sl.VIEW.VIEW_RIGHT)

        # Формирование кадра для работы в openCV
        self.image_color_left = mat_image_color_left.get_data()
        self.image_color_right = mat_image_color_right.get_data()

        return (self.image_color_left, self.image_color_right)

    def __saturation_point(self, point, shape):
        point[0] = point[0] if point[0] > 0 else 0
        point[0] = point[0] if point[0] < shape[0] else shape[0]

        point[1] = point[1] if point[1] > 0 else 0
        point[1] = point[1] if point[1] < shape[1] else shape[1]

        return point

    def _convert_MAT2IMAGE(self, point):
        (x, y) = point
        return (int(y), int(x))

    def _convert_IMAGE2MAT(self, point):
        (x, y) = point
        return (y, x)

    def _convert_Global2ROI(self, point, roi):
        x, y = point
        x += -roi[0][0]
        y += -roi[0][1]
        return (x, y)

    def _convert_ROI2Global(self, point, roi):
        x, y = point
        x += roi[0][0]
        y += roi[0][1]
        return (x, y)

    def _get_roi(self, roi_point, shape):
        (x, y) = roi_point
        roi_size = self.config["Detection"]["ROISize"]
        left_upper_point = self.__saturation_point([int(x - roi_size), int(y - roi_size)], shape)
        right_lower_point = self.__saturation_point([int(x + roi_size), int(y + roi_size)], shape)

        return [left_upper_point, right_lower_point]

    def _convert_Scaled2Global(self, point):
        (x, y) = point
        ratio = self.config["Detection"]["DownScaleRatio"]
        x = x * ratio
        y = y * ratio
        return (x, y)

    # Поиск минимальной точки
    def _get_minimum_point(self, image_hg):
        # Поиск минимума на кадре - это соответствует лучшей точки на кадре
        min_points = np.where(image_hg == np.amin(image_hg))

        # Транспонирование полученного массива в пары [(x,y), (x1,y1)]
        min_points = list(zip(min_points[0], min_points[1]))

        # Хрен знает что делать если точек несколько, просто берем первую
        return min_points[0]

    # Переход в новое цветовое пространство
    def _convert_BGR2HG(self, image):
        # Выделяем отдельные каналы из исходного изображения
        channels = cv2.split(image)

        # Переименование каналов
        red, blue, green = channels[2].astype(float), channels[0].astype(float), channels[1].astype(float)

        ball_red = float(self.ball_color[2])
        ball_blue = float(self.ball_color[0])
        ball_green = float(self.ball_color[1])

        # В зависимости от превалирующего канала выбираем формулу для расчёта
        if np.argmax(self.ball_color) == 0:
            ballrb = ball_red / ball_blue
            ballgb = ball_green / ball_blue

            blue = np.where(blue < self.config["Detection"]["ChannelThreshold"], 1, blue)
            colorspace_HG = abs(red / blue - ballrb) + abs(green / blue - ballgb)

        elif np.argmax(self.ball_color) == 1:
            ballrg = ball_red / ball_green
            ballbg = ball_blue / ball_green

            green = np.where(green < self.config["Detection"]["ChannelThreshold"], 1, green)
            colorspace_HG = abs(red / green - ballrg) + abs(blue / green - ballbg)
        else:
            ballgr = ball_green / ball_red
            ballbr = ball_blue / ball_red

            red = np.where(red < self.config["Detection"]["ChannelThreshold"], 1, red)
            colorspace_HG = abs(green / red - ballgr) + abs(blue / red - ballbr)

        return colorspace_HG

    def _binarization_HG(self, image, min_point):
        height, width = image.shape
        blank_image = np.zeros((height, width, 1), np.uint8)

        treshold = image[min_point] * self.config["Detection"]["ColorHGModificator"]\
            + self.config["Detection"]["ColorHGMinimal"]

        if treshold > self.config["Detection"]["BinarizationThreshold"]:
            treshold = self.config["Detection"]["BinarizationThreshold"]

        blank_image = np.where(image > treshold, 0, 255)
        return blank_image.astype(np.uint8)

    def _find_contours(self, img_gray):
        # Поиск всех контуров на изображении
        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) is 0:
            raise ValueError("No contours")

        return contours

    def _processing_image(self, image_gray):
        image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
        image_gray = cv2.addWeighted(image_gray, 1.5, image_gray, 1.5, 0)
        return image_gray

    def _get_enclosing_radius(self, contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return radius

    def _sort_contours(self, contours):
        # Создаем массив из кортежей индексов контуров, самих контуров
        # и поля для рейтинга каждого контура
        enum_contrours = [[i, ctr, 0] for i, ctr in enumerate(contours)]

        # Сортировка по величине окружности
        enum_contrours.sort(key=lambda x: self._get_enclosing_radius(x[1]), reverse=True)

        # Добавляем в вес третий параметр
        for index, elem in enumerate(enum_contrours):
            elem[2] += pow(index, 2)

        # Возвращаем контур у которого минимальный третий элемент
        enum_contrours.sort(key=itemgetter(2))
        return enum_contrours[0][1]

    def _get_center_moment(self, contour):
        M = cv2.moments(contour)
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]

        center_point = (center_x, center_y)
        return center_point

    def _get_center_color(self, image, mask):
        # Выделяем все точки которые входят в черно-белую маску после бинаризации
        interested_points = np.where(mask == 255)
        # Транспонирование полученного массива в пары [(x,y), (x1,y1)]
        interested_points = list(zip(interested_points[0], interested_points[1]))

        # Вычисляем все значения пикселей из изображения в цветовом пространстве HG
        # которые попали в маску
        values = [image[point] for point in interested_points]

        # Находим минимальное/максимальное значение и их разницу
        value_max, value_min = max(values), min(values)
        delta_value = value_max - value_min

        x_accumulator = 0
        y_accumulator = 0
        weight_accumulator = 0

        # Проход по точкам и вычисление весовых коэфициентов
        for point in interested_points:
            # Вычисление веса в приведении в диапазон 0.0 - 1.0
            weight_value = 1 - (image[point] - value_min) / (delta_value)
            x_accumulator += point[0] * weight_value
            y_accumulator += point[1] * weight_value
            weight_accumulator += weight_value

        center_point = (x_accumulator / weight_accumulator, y_accumulator / weight_accumulator)

        return center_point

    # Поиск мяча на отдельном кадре
    def find_ball(self, image):
        # Уменьшаем изображение для быстрого поиска максимума цвета
        ratio = self.config["Detection"]["DownScaleRatio"]
        image_small = cv2.resize(image, (0, 0), fx=1 / ratio, fy=1 / ratio,
                                 interpolation=cv2.INTER_NEAREST)

        # Переход из BGR в HG
        image_hg = self._convert_BGR2HG(image_small)

        # Поиск точки повышенного интереса
        point_scaled = self._get_minimum_point(image_hg)

        # Не забываем обратно привести координаты к реальным размерам
        point_global = self._convert_Scaled2Global(point_scaled)

        # Размечаем зону интереса вокруг точки интереса
        roi = self._get_roi(point_global, image.shape)

        # Выделяем необходимую область интереса уже без даунскейла
        image_roi = image[roi[0][0]:roi[1][0], roi[0][1]:roi[1][1]]
        image_hg_roi = self._convert_BGR2HG(image_roi)

        # Бинаризация изображения
        point_roi = self._convert_Global2ROI(point_global, roi)

        binary_roi = self._binarization_HG(image_hg_roi, point_roi)

        # Доп. обработка
        binary_roi = self._processing_image(binary_roi)

        # Поиск самого большого контура
        contours = self._find_contours(binary_roi)

        # Поиск лучшего контура
        contour = self._sort_contours(contours)

        # Определение центра масс
        if self.center_calculation == CenterCalculationMode.ByMoments:
            center_roi = self._convert_IMAGE2MAT(self._get_center_moment(contour))
        else:
            try:
                center_roi = self._get_center_color(image_hg_roi, binary_roi)
            except ZeroDivisionError as err:
                raise ValueError("Error in center calculation")

        cv2.drawContours(image, [contour], 0, (0, 0, 255), 3)
        cv2.circle(image, self._convert_MAT2IMAGE(point_global), 15, (0, 255, 0), 2)

        # Определяем площадь контура
        contour_area = cv2.contourArea(contour)

        return self._convert_ROI2Global(center_roi, roi), contour_area

    # Расчет расстояния
    def calculate_distance(self, point_left, point_right):
        (x_left, y_left) = point_left
        (x_right, y_right) = point_right

        # Находим расстояние между центрами левого и правого кадра
        stereo_difference = y_left - y_right

        # Если расстояние отрицательное, то выдаем ошибку
        if(stereo_difference <= 0):
            raise ValueError("Negative stereo difference")

        # Стереобаза между камерами. Берется из матрицы трансляции правой камеры
        # относительно левой
        camera_stereobase = self.calibration_parameters.T[0]
        distance = self.calibration_parameters.left_cam.fy * camera_stereobase / stereo_difference

        return distance

    def convert_image2world(self, point_image, depth_world):
        coordinate_image_x = point_image[0]
        coordinate_image_y = point_image[1]

        principal_point_x = self.calibration_parameters.left_cam.cy
        principal_point_y = self.calibration_parameters.left_cam.cx

        focal_length_x = self.calibration_parameters.left_cam.fy
        focal_length_y = self.calibration_parameters.left_cam.fx

        coordinate_world_z = depth_world
        coordinate_world_y = depth_world * \
            (coordinate_image_y - principal_point_y) / focal_length_y
        coordinate_world_x = -depth_world * \
            (coordinate_image_x - principal_point_x) / focal_length_x

        return np.array([coordinate_world_x, coordinate_world_y, coordinate_world_z])

    # Расчет расстояния до мяча через знание величины его видимой площади в мм
    # и величины его видимой площади в пикселях
    def get_distance_by_model(self, image_area):
        real_area = pi * self.config["Detection"]["RealBallRadius"] ** 2
        distance = self.calibration_parameters.left_cam.fx * sqrt(real_area) / sqrt(image_area)
        return distance

    # Фильтрация полученного расстояния
    def filter_distance(self, distance_stereo, ball_image_area):
        distance_model = self.get_distance_by_model(ball_image_area)

        if abs(distance_stereo - distance_model) > self.config["Detection"]["DistanceTreshold"]\
                or distance_stereo > self.config["Detection"]["DistanceMaximum"]\
                or distance_stereo < self.config["Detection"]["DistanceMinimum"]:
            raise ValueError("Bad distance, Stereo: {}, Model: {}".format(
                round(distance_stereo, 2), round(distance_model, 2)))

    # Сохранение стереопары
    def save_stereoimage(self, time, image_left, image_right, imtype):

        downscale = self.config["Detection"]["ImageSaveDownscale"] 

        local_time_ms = int((time - self.start_time) * 1000)
        stereo = np.concatenate((image_left, image_right), axis=1)
        stereo_resize = cv2.resize(stereo, (0, 0), fx= downscale, fy= downscale)

        cv2.imwrite(self.data_path + "/image" + "-{0}ms".format(local_time_ms) + "-" + imtype + ".bmp", stereo_resize)

    # Расчёт положения мяча в СК камеры на основе двух кадров
    def calculate_ball_position(self, image_left, image_right):

        # Копируем файлы что бы не испорить исходники
        # image_left = image_left_raw.copy()
        # image_right = image_right_raw.copy()

        current_time = time.time()
        self.delta_time = current_time - self.previous_time
        self.previous_time = current_time

        # Если изображения логируются, то сохраняем исходные фотографии
        # Надо это сделать до find_ball() - там может сработать исключение
        if self.config["Detection"]["ImageRawLogging"]:
            self.save_stereoimage(current_time, image_left, image_right, "raw")

        ball_point_left, ball_area_left = self.find_ball(image_left)
        ball_point_right, ball_area_right = self.find_ball(image_right)

        # Сохраняем точку на кадре для того что бы дебаг скорости вывести
        self.ball_point_left = ball_point_left

        # Расчет расстояния до объекта по двум точкам на кадре
        # print("Ball left: {}, ball right: {}".format(ball_point_left, ball_point_right))
        distance = self.calculate_distance(ball_point_left, ball_point_right)

        # Фильтрация расстояния до мячика на базе его модели
        self.filter_distance(distance, ball_area_left)

        # Расчет положения мячика относительно левой камеры в мировой СК
        # на основе расстояния до него в мировой СК
        position_camera = self.convert_image2world(ball_point_left, distance)

        position_robot = self.transform_point(np.array(position_camera))
        # last_coordinate = transfromPoint(calib_data * 1000.0, M, np.array(coordinate))

        # Если удалось получить положение шарика, то логируем его
        if self.config["Detection"]["TrajectoryLogging"]:
            local_time_ms = int((current_time - self.start_time) * 1000)
            write_string = "{0};".format(local_time_ms)
            # write_string += "{0};{1};{2};".format(position_camera[0], position_camera[1], position_camera[2])
            write_string += "{0};{1};{2};".format(position_robot[0], position_robot[1], position_robot[2])
            self.log_file.write(write_string + "\n")

        self.delta_detect_time = current_time - self.previous_detect_time
        self.previous_detect_time = current_time

        # Расчитываем скорость мяча по друм его известным точкам
        self.velocity = (position_camera - self.last_position_camera) / self.delta_detect_time
        self.last_position_camera = position_camera
        self.last_position_robot = position_robot

        # Вывод на экран отладочной информации
        self.draw_debug_info(image_left)

         # Если изображения логируются, то сохраняем обработанные фотографии
        if self.config["Detection"]["ImageRecLogging"]:
            self.save_stereoimage(current_time, image_left, image_right, "rec")

        return self.last_position_robot

    def draw_debug_info(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (0, 255, 0)
        lineType = 2

        cv2.putText(image, "FPS: {0}".format(round(1 / self.delta_time, 1)),
                    (image.shape[1] - 200, 60), font, fontScale, fontColor, lineType)

        cv2.putText(image, "Coordinate: ({0}, {1}, {2}) mm".format(round(self.last_position_robot[0], 1),
                                                                   round(self.last_position_robot[1], 1),
                                                                   round(self.last_position_robot[2], 1)),
                    (20, 90), font, fontScale, fontColor, lineType)

        x, y = self._convert_MAT2IMAGE(self.ball_point_left)
        ptr_start = (int(x), int(y))
        ptr_end = (int(x) + int(self.velocity[1] / 5), int(y) + int(-self.velocity[0] / 5))
        cv2.arrowedLine(image, ptr_start, ptr_end, (0, 0, 255), 5)

    def set_ball_color(self, point, image):
        self.ball_color = image[self._convert_IMAGE2MAT(point)][:3]
        # Подстраиваем конфиг под изменившийся цвет
        self.config["Detection"]["BallColor"] = self.ball_color.tolist() 
        self.updateConfig() 
        print("New color: {0}".format(self.ball_color))



    def updateConfig(self):
        if self.jsonFile is None:
            return

        self.jsonFile.close() # Close the JSON file

        try:
            dumps = json.dumps(self.config, sort_keys=True, indent=4)
            self.jsonFile = open(self.ConfigName, "w+")
            self.jsonFile.write(dumps)
            self.jsonFile.close()
        except TypeError as err:
            print(repr(err))
            print(self.config)

        self.jsonFile = open(self.ConfigName, "r", encoding='utf-8')

    # Создание матрицы преобразования Из СК камеры в СК робота
    def create_transform_matrix(self, calib_data):
        Rx = np.matrix(np.array([
            [1, 0, 0],
            [0, cos(calib_data[3]), -sin(calib_data[3])],
            [0, sin(calib_data[3]), cos(calib_data[3])]                
        ]))
        Ry = np.matrix(np.array([    
            [cos(calib_data[4]), 0, sin(calib_data[4])],
            [0, 1, 0],
            [-sin(calib_data[4]), 0, cos(calib_data[4])]
        ]))
        Rz = np.matrix(np.array([
            [cos(calib_data[5]), -sin(calib_data[5]), 0],
            [sin(calib_data[5]), cos(calib_data[5]), 0],
            [0, 0, 1]
        ]))
        tempM = np.matmul(Rz, Ry)
        M = np.matmul(tempM, Rx)
        matrixM = np.matrix(M)
        return matrixM
    
    # Преобразование координат мячика в СК робота
    def transform_point(self, point):
        # Переход в мм
        calib_data = self.calib_data * 1000

        R = point * self.tranform_matrix
        R = R.tolist()[0]
        R[0] = R[0] + calib_data[0]
        R[1] = R[1] + calib_data[1]
        R[2] = R[2] + calib_data[2]
        return R

