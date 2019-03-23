import cv2
import socket
import time
import balldetection as bd
import math
import struct

from solution import BallMovement

# Положение курсора на кадре
mouse_position = (0, 0)
cursor_is_changed = False


def mouse_callback(event, x, y, flags, param):
    global mouse_position, cursor_is_changed
    # При двойном клике получаем текущие положение курсора
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_position = (x, y)
        cursor_is_changed = True


class Point:
    def __init__(self, t, x, y, z):
        self.t = t
        self.x = x
        self.y = y
        self.z = z

def main():
    host = "192.168.1.11"
    port = 49180
    address = (host, port)

    # Создание обьекта распознавани
    try:
        detection = bd.BallDetection('config.json', bd.DetectMode.Real)
    # Перехват исключения если проблема с файлом конфигурации
    except IOError as e:
        print(repr(e))
        exit()
    # Перехват исключения если проблема с открыванием камеры
    except OSError as e:
        print(repr(e))
        exit()

    # Настройка события связанного с мышкой
    cv2.namedWindow('ZED')
    cv2.setMouseCallback('ZED', mouse_callback)

    # Создание экземпляра класса предсказания движения
    ballMovement = BallMovement()
    start_time = time.time()
    
    prev_ball_position = [0, 0, 0]
    ball_position = [0, 0, 0]

    camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Бесконечный цикл работы
    key = ''
    while key != 113:
        try:
            frame_left, frame_right = detection.grab_stereo()
        # Если кадр битый, то пишем об этом и пропускаем цикл
        except ValueError as error:
            # print(repr(error))
            continue

        try:
            ball_position = detection.calculate_ball_position(frame_left, frame_right)

            if(prev_ball_position != ball_position):
                point = ballMovement.pushPoint(Point(time.time() - start_time, ball_position[0] * 0.001, ball_position[1] * 0.001, ball_position[2] * 0.001))            
                if (point != None and not math.isnan(point[0])):
                    msg = struct.pack("dddd", point[0], point[1], point[2], point[3])
                    print("Send msg: {0}".format(point))
                    camera_socket.sendto(msg, (host, port))
                else:
                    print("############################", point)
            prev_ball_position = ball_position

        # Если не удалось расчитать координаты то пишем об этом
        # но цикл не пропускаем
        except ValueError as error:
            # print(error)
            pass

        # Печатаем на левом кадре отладочную инфу и выводим на экран
        cv2.imshow('ZED', frame_left)

        # Автоподстройка цвета мячика по двойному нажатию на цвет мячика на рисунке
        global cursor_is_changed, mouse_position
        if cursor_is_changed:
            detection.set_ball_color(mouse_position, frame_left)
            cursor_is_changed = False

        # Даем 1 сек на проверку нажатия кнопки
        key = cv2.waitKey(1)

    # Если выходим, то закрываем все окна
    cv2.destroyAllWindows()
    print("Finish")


if __name__ == "__main__":
    main()
